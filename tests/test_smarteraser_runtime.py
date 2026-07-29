from types import SimpleNamespace

from PIL import Image
import pytest
import torch


def _runtime_config(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    clip_dir = tmp_path / "clip"
    checkpoint_dir.mkdir()
    clip_dir.mkdir()
    (checkpoint_dir / "clip_mlp_weight.pth").write_bytes(b"local")
    return {
        "checkpoint_dir": checkpoint_dir,
        "clip_dir": clip_dir,
        "device": "cpu",
        "dtype": "float16",
        "num_inference_steps": 2,
        "guidance_scale": 1.5,
        "seed": 42,
        "prompt": "Remove the instance of object",
        "negative_prompt": "",
    }


def test_runtime_rejects_missing_local_checkpoint(tmp_path):
    from backend.models.background_inpainting.smarteraser.runtime import (
        SmartEraserRuntime,
    )

    clip_dir = tmp_path / "clip"
    clip_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        SmartEraserRuntime(
            checkpoint_dir=tmp_path / "missing",
            clip_dir=clip_dir,
            device="cpu",
            dtype="float16",
            num_inference_steps=1,
            guidance_scale=1.5,
            seed=42,
            prompt="Remove the instance of object",
            negative_prompt="",
        )


def test_runtime_rejects_missing_clip_mlp_weight(tmp_path):
    from backend.models.background_inpainting.smarteraser.runtime import (
        SmartEraserRuntime,
    )

    checkpoint_dir = tmp_path / "checkpoint"
    clip_dir = tmp_path / "clip"
    checkpoint_dir.mkdir()
    clip_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="clip_mlp_weight"):
        SmartEraserRuntime(
            checkpoint_dir=checkpoint_dir,
            clip_dir=clip_dir,
            device="cpu",
            dtype="float32",
            num_inference_steps=1,
            guidance_scale=1.5,
            seed=42,
            prompt="Remove the instance of object",
            negative_prompt="",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dtype", "float64", "dtype"),
        ("num_inference_steps", 0, "steps"),
        ("guidance_scale", 0.0, "guidance"),
    ],
)
def test_runtime_rejects_invalid_inference_config(
    tmp_path,
    field,
    value,
    message,
):
    from backend.models.background_inpainting.smarteraser.runtime import (
        SmartEraserRuntime,
    )

    config = _runtime_config(tmp_path)
    config[field] = value

    with pytest.raises(ValueError, match=message):
        SmartEraserRuntime(**config)


def test_runtime_loads_local_resources_and_wires_inference(
    monkeypatch,
    tmp_path,
):
    from backend.models.background_inpainting.smarteraser import runtime

    events = {}

    class FakeModule:
        def __init__(self, name):
            self.name = name

        def eval(self):
            events[f"{self.name}_eval"] = True
            return self

        def to(self, *args, **kwargs):
            events[f"{self.name}_to"] = (args, kwargs)
            return self

    class FakeClip:
        def __init__(self, path):
            events["clip_path"] = path
            self.vision_model = FakeModule("vision")
            self.text_model = FakeModule("text")
            self.clip_mlp = FakeModule("mlp")

        def load_mlp_weight(self, path):
            events["mlp_path"] = path

        def inference_vtoken(
            self,
            input_ids,
            negative_ids,
            image,
            text_encoder,
        ):
            events["visual_token"] = (
                input_ids.clone(),
                negative_ids.clone(),
                image.clone(),
                text_encoder,
            )
            return torch.ones((1, 7, 4)), torch.zeros((1, 7, 4))

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            events["processor_load"] = (path, kwargs)
            return cls()

        def __call__(self, *, images, return_tensors):
            events["guidance_image"] = images.copy()
            assert return_tensors == "pt"
            return {"pixel_values": torch.ones((1, 3, 2, 2))}

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            events["tokenizer_load"] = (path, kwargs)
            return cls()

        def __call__(self, text, **kwargs):
            events.setdefault("prompts", []).append((text, kwargs))
            token = 1 if text else 0
            return {"input_ids": torch.full((1, 7), token)}

    class FakePipeline:
        def __init__(self):
            self.text_encoder = object()
            self.device = torch.device("cpu")

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            events["pipeline_load"] = (path, kwargs)
            return cls()

        def to(self, device):
            events["pipeline_device"] = device
            self.device = torch.device(device)
            return self

        def __call__(self, **kwargs):
            events["pipeline_call"] = kwargs
            return SimpleNamespace(images=[Image.new("RGBA", (8, 8), "red")])

    monkeypatch.setattr(runtime, "CLIPVisualPrompt", FakeClip)
    monkeypatch.setattr(runtime, "CLIPImageProcessor", FakeProcessor)
    monkeypatch.setattr(runtime, "CLIPTokenizer", FakeTokenizer)
    monkeypatch.setattr(
        runtime,
        "StableDiffusionInpaintRegionPipeline",
        FakePipeline,
    )

    config = _runtime_config(tmp_path)
    model = runtime.SmartEraserRuntime(**config)
    image = Image.new("RGB", (8, 8), "blue")
    mask = Image.new("L", image.size, 0)
    mask.paste(255, (3, 3, 5, 5))

    result = model.inpaint(image, mask)

    assert result.mode == "RGB"
    assert result.size == image.size
    assert events["pipeline_load"] == (
        str(config["checkpoint_dir"].resolve()),
        {
            "torch_dtype": torch.float32,
            "local_files_only": True,
        },
    )
    assert events["processor_load"][0] == str(config["clip_dir"].resolve())
    assert events["processor_load"][1] == {"local_files_only": True}
    assert events["tokenizer_load"][1] == {"local_files_only": True}
    assert events["mlp_path"] == str(
        (config["checkpoint_dir"] / "clip_mlp_weight.pth").resolve()
    )
    assert events["prompts"][0][0] == config["prompt"]
    assert events["prompts"][1][0] == config["negative_prompt"]
    call = events["pipeline_call"]
    assert call["image"] is image
    assert call["mask_image"] is mask
    assert call["num_inference_steps"] == 2
    assert call["guidance_scale"] == 1.5
    assert call["generator"].initial_seed() == 42


def test_runtime_close_releases_owned_resources(monkeypatch, tmp_path):
    from backend.models.background_inpainting.smarteraser import runtime

    class FakeClip:
        def __init__(self, _path):
            self.vision_model = SimpleNamespace(
                eval=lambda: self.vision_model,
                to=lambda *args, **kwargs: self.vision_model,
            )
            self.text_model = SimpleNamespace(
                eval=lambda: self.text_model,
                to=lambda *args, **kwargs: self.text_model,
            )
            self.clip_mlp = SimpleNamespace(
                eval=lambda: self.clip_mlp,
                to=lambda *args, **kwargs: self.clip_mlp,
            )

        def load_mlp_weight(self, _path):
            return None

    class FakeLoader:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def to(self, _device):
            self.text_encoder = object()
            return self

    monkeypatch.setattr(runtime, "CLIPVisualPrompt", FakeClip)
    monkeypatch.setattr(runtime, "CLIPImageProcessor", FakeLoader)
    monkeypatch.setattr(runtime, "CLIPTokenizer", FakeLoader)
    monkeypatch.setattr(
        runtime,
        "StableDiffusionInpaintRegionPipeline",
        FakeLoader,
    )
    model = runtime.SmartEraserRuntime(**_runtime_config(tmp_path))

    model.close()

    assert model.pipeline is None
    assert model.clip_model is None
    assert model.clip_processor is None
    assert model.tokenizer is None
