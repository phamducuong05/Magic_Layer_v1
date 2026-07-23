"""Application adapter for HD-Painter hidden-object reconstruction."""

from __future__ import annotations

import gc
import importlib
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

from ...core.logging import get_logger, log_event, trace_stage
from ..base import BaseObjectReconstructionModel
from ..registry import ModelRegistry

logger = get_logger(__name__)


class ObjectReconstructionError(RuntimeError):
    """Raised when HD-Painter cannot produce an application-valid crop."""

    def __init__(self, message: str, *, stage: str = "inference"):
        super().__init__(message)
        self.stage = stage


def _move_ddim_model(model: Any, device: str) -> None:
    """Move every module owned by the research DDIM container."""
    for name in ("vae", "encoder", "unet", "low_scale_model"):
        module = getattr(model, name, None)
        if module is not None and hasattr(module, "to"):
            module.to(device=device)
    encoder = getattr(model, "encoder", None)
    if encoder is not None and hasattr(encoder, "device"):
        encoder.device = device


def _offload_ddim_model(model: Any) -> None:
    """Drop gradients and move one DDIM model out of CUDA memory."""
    unet = getattr(model, "unet", None)
    if unet is not None:
        if hasattr(unet, "zero_grad"):
            unet.zero_grad(set_to_none=True)
        if hasattr(unet, "requires_grad_"):
            unet.requires_grad_(False)
    _move_ddim_model(model, "cpu")


def _release_cuda_cache() -> None:
    """Release inactive allocations at the measured stage boundary."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass(frozen=True)
class _Runtime:
    IImage: type
    load_inpainting_model: Callable[..., Any]
    load_sr_model: Callable[..., Any] | None
    sd_run: Callable[..., Any]
    rasg_run: Callable[..., Any]
    sr_run: Callable[..., Any] | None
    reset_state: Callable[[], None]
    clear_model_cache: Callable[[], None]


@dataclass
class _PreparedRequest:
    """Normalized inputs and intermediate output for one reconstruction."""

    source: Image.Image
    hard_mask: Image.Image
    low_image: Image.Image
    low_mask: Image.Image
    prompt: str
    generated: Image.Image | None = None


def _reset_research_state(
    router: ModuleType, share: ModuleType, painta: ModuleType
) -> None:
    """Clear mutable inference state while preserving cached model weights."""
    router.reset()
    painta.painta_on = False
    painta.token_idx = []
    for name in (
        "input_mask",
        "input_shape",
        "timestep",
        "timestep_index",
        "painta_mask",
        "mask8",
        "mask16",
        "mask32",
        "mask64",
        "_crossattn_similarity_res8",
        "_crossattn_similarity_res16",
        "_crossattn_similarity_res32",
        "_crossattn_similarity_res64",
    ):
        if hasattr(share, name):
            setattr(share, name, None)


def _load_runtime(
    *,
    checkpoint_root: Path,
    enable_super_resolution: bool,
    auto_download: bool,
) -> _Runtime:
    """Import HD-Painter only after the manager requests the adapter."""
    package = "backend.models.object_reconstruction.hd_painter.src"
    try:
        common = importlib.import_module(f"{package}.models.common")
        inpainting = importlib.import_module(f"{package}.models.inpainting")
        sd_method = importlib.import_module(f"{package}.methods.sd")
        rasg_method = importlib.import_module(f"{package}.methods.rasg")
        iimage = importlib.import_module(f"{package}.utils.iimage")
        router = importlib.import_module(
            f"{package}.smplfusion.patches.router"
        )
        share = importlib.import_module(f"{package}.smplfusion.share")
        painta = importlib.import_module(
            f"{package}.smplfusion.patches.attentionpatch.painta"
        )

        common.MODEL_FOLDER = str(checkpoint_root)
        common.AUTO_DOWNLOAD_ALLOWED = auto_download
        inpainting.MODEL_FOLDER = str(checkpoint_root)

        load_sr_model = None
        sr_run = None
        if enable_super_resolution:
            sr_model = importlib.import_module(f"{package}.models.sd2_sr")
            sr_method = importlib.import_module(f"{package}.methods.sr")
            sr_model.MODEL_FOLDER = str(checkpoint_root)
            sr_model.MODEL_PATH = str(
                checkpoint_root
                / "sd-2-0-upsample"
                / "x4-upscaler-ema.safetensors"
            )
            load_sr_model = sr_model.load_model
            sr_run = sr_method.run
    except Exception as exc:
        raise ObjectReconstructionError(
            "Unable to import HD-Painter inference dependencies. Install the "
            "object-reconstruction requirements without downgrading the main "
            f"backend environment. Original error: {exc}"
        ) from exc

    return _Runtime(
        IImage=iimage.IImage,
        load_inpainting_model=inpainting.load_inpainting_model,
        load_sr_model=load_sr_model,
        sd_run=sd_method.run,
        rasg_run=rasg_method.run,
        sr_run=sr_run,
        reset_state=lambda: _reset_research_state(router, share, painta),
        clear_model_cache=inpainting.model_cache.clear,
    )


@ModelRegistry.register("object_reconstruction", "hd_painter")
class HDPainterObjectReconstruction(BaseObjectReconstructionModel):
    """Run HD-Painter on one square RGB crop and aligned hard mask."""

    _inference_lock = threading.Lock()
    _MODEL_IDS = {"ds8_inp", "sd15_inp", "sd2_inp"}
    _METHODS = {"baseline", "painta", "rasg", "painta+rasg"}

    def _load_model(self) -> None:
        self._settings = self._validated_settings(self.config)
        self._last_debug_artifacts: list[dict[str, Image.Image]] = []
        if not str(self.device).startswith("cuda") or not torch.cuda.is_available():
            raise ObjectReconstructionError(
                "HD-Painter requires CUDA; configured device is "
                f"{self.device!r} and torch.cuda.is_available() is "
                f"{torch.cuda.is_available()}."
            )

        source_root = Path(__file__).resolve().parent / "hd_painter"
        checkpoint_root = Path(self._settings["checkpoint_root"])
        if not checkpoint_root.is_absolute():
            checkpoint_root = source_root / checkpoint_root
        checkpoint_root = checkpoint_root.resolve()
        checkpoint_root.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initializing HD-Painter from %s with checkpoints at %s",
            source_root,
            checkpoint_root,
        )
        self._runtime = _load_runtime(
            checkpoint_root=checkpoint_root,
            enable_super_resolution=self._settings["super_resolution"][
                "enabled"
            ],
            auto_download=self._settings["auto_download"],
        )
        dtype = torch.float16 if self._settings["fp16"] else torch.float32
        load_device = (
            "cpu"
            if self._settings["sequential_cpu_offload"]
            else self.device
        )
        try:
            self._inpainting_model = self._runtime.load_inpainting_model(
                model_id=self._settings["model_id"],
                dtype=dtype,
                device=load_device,
                cache=True,
            )
            if self._settings["sequential_cpu_offload"]:
                # The research cache ignores the requested device when it
                # returns an existing model, so enforce CPU residency here.
                _offload_ddim_model(self._inpainting_model)
                _release_cuda_cache()
            self._sr_model = None
            if self._settings["super_resolution"]["enabled"]:
                if self._runtime.load_sr_model is None:
                    raise ObjectReconstructionError(
                        "HD-Painter super-resolution runtime is unavailable."
                    )
                self._sr_model = self._runtime.load_sr_model(
                    dtype=dtype,
                    device=load_device,
                )
                if self._settings["sequential_cpu_offload"]:
                    _offload_ddim_model(self._sr_model)
                    _release_cuda_cache()
        except ObjectReconstructionError:
            raise
        except Exception as exc:
            raise ObjectReconstructionError(
                f"Failed to initialize HD-Painter weights: {exc}"
            ) from exc

    @classmethod
    def _validated_settings(cls, raw: dict[str, Any]) -> dict[str, Any]:
        sr_raw = dict(raw.get("super_resolution", {}))
        settings = {
            "model_id": raw.get("model_id", "ds8_inp"),
            "method": raw.get("method", "painta+rasg"),
            "input_size": int(raw.get("input_size", 512)),
            "num_steps": int(raw.get("num_steps", 50)),
            "guidance_scale": float(raw.get("guidance_scale", 7.5)),
            "rasg_eta": float(raw.get("rasg_eta", 0.1)),
            "seed": int(raw.get("seed", 1)),
            "positive_prompt": str(raw.get("positive_prompt", "")),
            "negative_prompt": str(raw.get("negative_prompt", "")),
            "fp16": bool(raw.get("fp16", True)),
            "sequential_cpu_offload": bool(
                raw.get("sequential_cpu_offload", True)
            ),
            "auto_download": bool(raw.get("auto_download", True)),
            "checkpoint_root": str(raw.get("checkpoint_root", "checkpoints")),
            "super_resolution": {
                "enabled": bool(sr_raw.get("enabled", True)),
                "minimum_roi_size": int(
                    sr_raw.get("minimum_roi_size", 0)
                ),
                "target_size": int(sr_raw.get("target_size", 2048)),
                "noise_level": int(sr_raw.get("noise_level", 20)),
                "denoising_stride": int(
                    sr_raw.get("denoising_stride", 50)
                ),
                "guidance_scale": float(
                    sr_raw.get("guidance_scale", 7.5)
                ),
                "blend_trick": bool(sr_raw.get("blend_trick", True)),
                "blend_output": bool(sr_raw.get("blend_output", True)),
                "use_sam_mask": bool(sr_raw.get("use_sam_mask", False)),
                "prompt_suffix": str(sr_raw.get("prompt_suffix", "")),
            },
        }
        if settings["model_id"] not in cls._MODEL_IDS:
            raise ObjectReconstructionError(
                f"Invalid HD-Painter model_id: {settings['model_id']!r}."
            )
        if settings["method"] not in cls._METHODS:
            raise ObjectReconstructionError(
                f"Invalid HD-Painter method: {settings['method']!r}."
            )
        if settings["input_size"] != 512:
            raise ObjectReconstructionError("HD-Painter input_size must be 512.")
        if not 1 <= settings["num_steps"] <= 1000:
            raise ObjectReconstructionError(
                "HD-Painter num_steps must be between 1 and 1000."
            )
        if 1000 // settings["num_steps"] <= 0:
            raise ObjectReconstructionError(
                "HD-Painter num_steps produces an invalid DDIM stride."
            )
        if settings["guidance_scale"] <= 0:
            raise ObjectReconstructionError(
                "HD-Painter guidance_scale must be positive."
            )
        sr = settings["super_resolution"]
        if sr["minimum_roi_size"] < 0:
            raise ObjectReconstructionError(
                "HD-Painter super_resolution.minimum_roi_size must be "
                "non-negative."
            )
        if sr["enabled"] and sr["use_sam_mask"]:
            raise ObjectReconstructionError(
                "HD-Painter use_sam_mask must remain false; the pipeline "
                "already supplies the validated reconstruction mask."
            )
        if sr["enabled"] and sr["target_size"] != 2048:
            raise ObjectReconstructionError(
                "HD-Painter super-resolution target_size must be 2048."
            )
        if sr["enabled"] and not 1 <= sr["denoising_stride"] <= 999:
            raise ObjectReconstructionError(
                "HD-Painter super-resolution denoising_stride must be in "
                "the range 1..999."
            )
        if sr["enabled"] and sr["guidance_scale"] <= 0:
            raise ObjectReconstructionError(
                "HD-Painter super-resolution guidance_scale must be positive."
            )
        return settings

    def reconstruct(
        self,
        image: Image.Image,
        mask: Image.Image,
        object_context: str = "",
    ) -> Image.Image:
        """Reconstruct one crop through the same phase scheduler as batches."""
        outcome = self.reconstruct_many([(image, mask, object_context)])[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def _prepare_request(
        self,
        image: Image.Image,
        mask: Image.Image,
        object_context: str,
    ) -> _PreparedRequest | Image.Image:
        """Normalize one request, returning source directly for an empty mask."""
        source = image.convert("RGB")
        original_size = source.size
        hard_mask = mask.convert("L")
        if hard_mask.size != original_size:
            hard_mask = hard_mask.resize(
                original_size, Image.Resampling.NEAREST
            )
        hard_mask = Image.fromarray(
            np.where(np.asarray(hard_mask) >= 127, 255, 0).astype(np.uint8),
            mode="L",
        )
        if not np.any(np.asarray(hard_mask)):
            return source

        prompt = object_context.strip()
        if not prompt:
            raise ObjectReconstructionError(
                "HD-Painter requires non-empty object_context."
            )

        size = self._settings["input_size"]
        low_image = source.resize((size, size), Image.Resampling.LANCZOS)
        low_mask = hard_mask.resize((size, size), Image.Resampling.NEAREST)
        low_mask = Image.fromarray(
            np.where(np.asarray(low_mask) >= 127, 255, 0).astype(np.uint8),
            mode="L",
        ).convert("RGB")

        return _PreparedRequest(
            source=source,
            hard_mask=hard_mask,
            low_image=low_image,
            low_mask=low_mask,
            prompt=prompt,
        )

    @staticmethod
    def _stage_error(exc: Exception, stage: str) -> ObjectReconstructionError:
        """Attach the failing HD-Painter phase without losing explicit stages."""
        if (
            isinstance(exc, ObjectReconstructionError)
            and exc.stage != "inference"
        ):
            return exc
        return ObjectReconstructionError(
            f"HD-Painter {stage} failed: {exc}",
            stage=stage,
        )

    def reconstruct_many(
        self,
        requests: Sequence[tuple[Image.Image, Image.Image, str]],
    ) -> list[Image.Image | Exception]:
        """Run all 512px generations, then all eligible SR jobs."""
        outcomes: list[Image.Image | Exception | None] = [None] * len(requests)
        self._last_debug_artifacts = [{} for _ in requests]
        prepared: dict[int, _PreparedRequest] = {}
        for index, (image, mask, object_context) in enumerate(requests):
            try:
                normalized = self._prepare_request(
                    image, mask, object_context
                )
            except Exception as exc:
                outcomes[index] = self._stage_error(
                    exc, "input_preparation"
                )
                continue
            if isinstance(normalized, Image.Image):
                outcomes[index] = normalized
            else:
                prepared[index] = normalized

        if not prepared:
            return [outcome for outcome in outcomes if outcome is not None]

        with self._inference_lock:
            sr_phase_started = False
            try:
                if self._settings["sequential_cpu_offload"]:
                    _move_ddim_model(self._inpainting_model, self.device)
                runner = (
                    self._runtime.rasg_run
                    if self._settings["method"] in {"rasg", "painta+rasg"}
                    else self._runtime.sd_run
                )
                for index, item in prepared.items():
                    try:
                        with trace_stage(
                            logger,
                            "hd_painter_generation_512",
                            request_index=index,
                            method=self._settings["method"],
                            input_size=item.low_image.size,
                            num_steps=self._settings["num_steps"],
                        ):
                            generated = runner(
                                ddim=self._inpainting_model,
                                method=self._settings["method"],
                                prompt=item.prompt,
                                image=self._runtime.IImage(item.low_image),
                                mask=self._runtime.IImage(item.low_mask),
                                seed=self._settings["seed"],
                                eta=self._settings["rasg_eta"],
                                negative_prompt=self._settings[
                                    "negative_prompt"
                                ],
                                positive_prompt=self._settings[
                                    "positive_prompt"
                                ],
                                num_steps=self._settings["num_steps"],
                                guidance_scale=self._settings[
                                    "guidance_scale"
                                ],
                            )
                        item.generated = self._to_single_pil(generated)
                        self._last_debug_artifacts[index][
                            "base_output_512"
                        ] = item.generated.copy()
                    except Exception as exc:
                        outcomes[index] = self._stage_error(
                            exc, "generation_512"
                        )
                    finally:
                        # RASG and PAINTA store attention state globally.
                        self._runtime.reset_state()

                sr = self._settings["super_resolution"]
                sr_indices = [
                    index
                    for index, item in prepared.items()
                    if outcomes[index] is None
                    and item.generated is not None
                    and sr["enabled"]
                    and max(item.source.size) >= sr["minimum_roi_size"]
                ]
                if self._settings["sequential_cpu_offload"]:
                    _offload_ddim_model(self._inpainting_model)
                    _release_cuda_cache()

                if sr_indices:
                    if self._runtime.sr_run is None or self._sr_model is None:
                        error = ObjectReconstructionError(
                            "HD-Painter super-resolution was enabled but not loaded."
                        )
                        for index in sr_indices:
                            outcomes[index] = self._stage_error(
                                error, "super_resolution"
                            )
                        sr_indices = []
                    if self._settings["sequential_cpu_offload"]:
                        if sr_indices:
                            _move_ddim_model(self._sr_model, self.device)
                    sr_phase_started = bool(sr_indices)
                    for index in sr_indices:
                        item = prepared[index]
                        sr_prompt = item.prompt
                        if sr["prompt_suffix"]:
                            sr_prompt = (
                                f"{item.prompt}, {sr['prompt_suffix']}"
                            )
                        try:
                            with trace_stage(
                                logger,
                                "hd_painter_super_resolution",
                                request_index=index,
                                source_size=item.source.size,
                                target_size=sr["target_size"],
                            ):
                                outcomes[index] = self._to_single_pil(
                                    self._runtime.sr_run(
                                        ddim=self._sr_model,
                                        sam_predictor=None,
                                        lr_image=item.generated,
                                        hr_image=item.source,
                                        hr_mask=item.hard_mask.convert("RGB"),
                                        prompt=sr_prompt,
                                        noise_level=sr["noise_level"],
                                        blend_output=sr["blend_output"],
                                        blend_trick=sr["blend_trick"],
                                        dt=sr["denoising_stride"],
                                        seed=self._settings["seed"],
                                        guidance_scale=sr["guidance_scale"],
                                        negative_prompt=self._settings[
                                            "negative_prompt"
                                        ],
                                        use_sam_mask=False,
                                    )
                                )
                                self._last_debug_artifacts[index][
                                    "sr_output"
                                ] = outcomes[index].copy()
                        except Exception as exc:
                            outcomes[index] = self._stage_error(
                                exc, "super_resolution"
                            )

                for index, item in prepared.items():
                    if outcomes[index] is not None:
                        continue
                    if item.generated is None:
                        outcomes[index] = ObjectReconstructionError(
                            "HD-Painter generation produced no image.",
                            stage="generation_512",
                        )
                        continue
                    outcomes[index] = item.generated
                    if sr["enabled"]:
                        log_event(
                            logger,
                            "hd_painter_super_resolution",
                            "skip",
                            request_index=index,
                            roi_size=max(item.source.size),
                            minimum_roi_size=sr["minimum_roi_size"],
                            reason="roi_below_threshold",
                        )
            finally:
                if sr_phase_started:
                    self._runtime.reset_state()
                if self._settings["sequential_cpu_offload"]:
                    _offload_ddim_model(self._inpainting_model)
                    if self._sr_model is not None:
                        _offload_ddim_model(self._sr_model)
                    _release_cuda_cache()

        for index, item in prepared.items():
            result = outcomes[index]
            if isinstance(result, Exception):
                continue
            try:
                outcomes[index] = result.convert("RGB").resize(
                    item.source.size, Image.Resampling.LANCZOS
                )
            except Exception as exc:
                outcomes[index] = ObjectReconstructionError(
                    f"HD-Painter crop-size restoration failed: {exc}",
                    stage="crop_size_restoration",
                )

        return [
            outcome
            if outcome is not None
            else ObjectReconstructionError("Missing reconstruction outcome.")
            for outcome in outcomes
        ]

    def consume_debug_artifacts(self) -> list[dict[str, Image.Image]]:
        """Return and clear intermediate images from the latest batch."""
        artifacts = self._last_debug_artifacts
        self._last_debug_artifacts = []
        return artifacts

    def offload_to_cpu(self) -> None:
        """Free CUDA residency while retaining reusable model weights in RAM."""
        with self._inference_lock:
            self._runtime.reset_state()
            _offload_ddim_model(self._inpainting_model)
            if self._sr_model is not None:
                _offload_ddim_model(self._sr_model)
            _release_cuda_cache()

    def unload(self) -> None:
        """Remove HD-Painter weights, including its module-level SD cache."""
        with self._inference_lock:
            runtime = getattr(self, "_runtime", None)
            if runtime is not None:
                runtime.reset_state()
            inpainting_model = getattr(self, "_inpainting_model", None)
            if inpainting_model is not None:
                _offload_ddim_model(inpainting_model)
            sr_model = getattr(self, "_sr_model", None)
            if sr_model is not None:
                _offload_ddim_model(sr_model)
            if runtime is not None:
                runtime.clear_model_cache()
            super().unload()
            _release_cuda_cache()

    @staticmethod
    def _to_single_pil(value: Any) -> Image.Image:
        if hasattr(value, "pil") and callable(value.pil):
            value = value.pil()
        if isinstance(value, (list, tuple)):
            raise ObjectReconstructionError(
                "HD-Painter returned multiple samples; exactly one is required."
            )
        if not isinstance(value, Image.Image):
            raise ObjectReconstructionError(
                "HD-Painter returned a non-image result."
            )
        return value.convert("RGB")
