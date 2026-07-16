"""Application adapter for HD-Painter hidden-object reconstruction."""

from __future__ import annotations

import importlib
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

from ..base import BaseObjectReconstructionModel
from ..registry import ModelRegistry

logger = logging.getLogger(__name__)


class ObjectReconstructionError(RuntimeError):
    """Raised when HD-Painter cannot produce an application-valid crop."""


@dataclass(frozen=True)
class _Runtime:
    IImage: type
    load_inpainting_model: Callable[..., Any]
    load_sr_model: Callable[..., Any] | None
    sd_run: Callable[..., Any]
    rasg_run: Callable[..., Any]
    sr_run: Callable[..., Any] | None
    reset_state: Callable[[], None]


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
    )


@ModelRegistry.register("object_reconstruction", "hd_painter")
class HDPainterObjectReconstruction(BaseObjectReconstructionModel):
    """Run HD-Painter on one square RGB crop and aligned hard mask."""

    _inference_lock = threading.Lock()
    _MODEL_IDS = {"ds8_inp", "sd15_inp", "sd2_inp"}
    _METHODS = {"baseline", "painta", "rasg", "painta+rasg"}

    def _load_model(self) -> None:
        self._settings = self._validated_settings(self.config)
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
        try:
            self._inpainting_model = self._runtime.load_inpainting_model(
                model_id=self._settings["model_id"],
                dtype=dtype,
                device=self.device,
                cache=True,
            )
            self._sr_model = None
            if self._settings["super_resolution"]["enabled"]:
                if self._runtime.load_sr_model is None:
                    raise ObjectReconstructionError(
                        "HD-Painter super-resolution runtime is unavailable."
                    )
                self._sr_model = self._runtime.load_sr_model(
                    dtype=dtype,
                    device=self.device,
                )
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
            "auto_download": bool(raw.get("auto_download", True)),
            "checkpoint_root": str(raw.get("checkpoint_root", "checkpoints")),
            "super_resolution": {
                "enabled": bool(sr_raw.get("enabled", True)),
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

        with self._inference_lock:
            try:
                runner = (
                    self._runtime.rasg_run
                    if self._settings["method"] in {"rasg", "painta+rasg"}
                    else self._runtime.sd_run
                )
                generated = runner(
                    ddim=self._inpainting_model,
                    method=self._settings["method"],
                    prompt=prompt,
                    image=self._runtime.IImage(low_image),
                    mask=self._runtime.IImage(low_mask),
                    seed=self._settings["seed"],
                    eta=self._settings["rasg_eta"],
                    negative_prompt=self._settings["negative_prompt"],
                    positive_prompt=self._settings["positive_prompt"],
                    num_steps=self._settings["num_steps"],
                    guidance_scale=self._settings["guidance_scale"],
                )
                result = self._to_single_pil(generated)

                sr = self._settings["super_resolution"]
                if sr["enabled"]:
                    if self._runtime.sr_run is None or self._sr_model is None:
                        raise ObjectReconstructionError(
                            "HD-Painter super-resolution was enabled but not loaded."
                        )
                    sr_prompt = prompt
                    if sr["prompt_suffix"]:
                        sr_prompt = f"{prompt}, {sr['prompt_suffix']}"
                    result = self._to_single_pil(
                        self._runtime.sr_run(
                            ddim=self._sr_model,
                            sam_predictor=None,
                            lr_image=self._runtime.IImage(result),
                            hr_image=self._runtime.IImage(source),
                            hr_mask=self._runtime.IImage(
                                hard_mask.convert("RGB")
                            ),
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
            except ObjectReconstructionError:
                raise
            except Exception as exc:
                raise ObjectReconstructionError(
                    f"HD-Painter inference failed: {exc}"
                ) from exc
            finally:
                self._runtime.reset_state()

        return result.convert("RGB").resize(
            original_size, Image.Resampling.LANCZOS
        )

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
