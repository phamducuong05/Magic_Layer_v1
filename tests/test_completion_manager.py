"""Tests for lazy completion-model lifecycle management."""

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import yaml


MODELS_PATH = Path(__file__).resolve().parents[1] / "backend" / "models"
models_package = types.ModuleType("backend.models")
models_package.__path__ = [str(MODELS_PATH)]
sys.modules["backend.models"] = models_package

for category, adapter_names in {
    "segmentation": ["sam3"],
    "matting": ["birefnet"],
    "background_inpainting": [],
}.items():
    package_name = f"backend.models.{category}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(MODELS_PATH / category)]
    sys.modules[package_name] = package
    for adapter_name in adapter_names:
        module_name = f"{package_name}.{adapter_name}"
        adapter = types.ModuleType(module_name)
        sys.modules[module_name] = adapter
        setattr(package, adapter_name, adapter)

manager_module = importlib.import_module("backend.models.manager")


def _new_manager():
    manager_module.ModelManager._instance = None
    return manager_module.ModelManager()


def test_completion_model_is_created_lazily_and_reused(monkeypatch):
    created = []

    class FakeCompletionAdapter:
        def __init__(self, config, device):
            created.append((config, device))

    fake_config = Mock()
    fake_config.device = "cuda:1"
    fake_config.get_model_config.return_value = {
        "name": "sdamodal",
        "input_size": 512,
    }
    monkeypatch.setattr(manager_module, "config", fake_config)
    get_class = Mock(return_value=FakeCompletionAdapter)
    monkeypatch.setattr(manager_module.ModelRegistry, "get_class", get_class)

    manager = _new_manager()

    assert manager._completion_model is None
    first = manager.get_completion_model()
    second = manager.get_completion_model()

    assert first is second
    assert created == [({"input_size": 512}, "cuda:1")]
    get_class.assert_called_once_with("completion", "sdamodal")


def test_warmup_does_not_load_completion(monkeypatch):
    manager = _new_manager()
    monkeypatch.setattr(manager, "get_segmentation_model", Mock())
    monkeypatch.setattr(manager, "get_matting_model", Mock())
    monkeypatch.setattr(manager, "get_background_inpainting_model", Mock())
    completion_getter = Mock()
    object_reconstruction_getter = Mock()
    monkeypatch.setattr(
        manager,
        "get_completion_model",
        completion_getter,
        raising=False,
    )
    monkeypatch.setattr(
        manager,
        "get_object_reconstruction_model",
        object_reconstruction_getter,
    )

    manager.warmup_all()

    completion_getter.assert_not_called()
    object_reconstruction_getter.assert_not_called()


def test_release_model_unloads_instance_and_drops_manager_reference(
    monkeypatch,
):
    manager = _new_manager()
    model = Mock()
    manager._completion_model = model
    release_cuda_memory = Mock()
    monkeypatch.setattr(
        manager_module,
        "_release_cuda_memory",
        release_cuda_memory,
        raising=False,
    )

    released = manager.release_model("completion")

    assert released is True
    model.unload.assert_called_once_with()
    assert manager._completion_model is None
    release_cuda_memory.assert_called_once_with()


def test_offload_model_keeps_instance_available_for_next_request(monkeypatch):
    manager = _new_manager()
    model = Mock()
    manager._object_reconstruction_model = model
    release_cuda_memory = Mock()
    monkeypatch.setattr(
        manager_module,
        "_release_cuda_memory",
        release_cuda_memory,
        raising=False,
    )

    offloaded = manager.offload_model("object_reconstruction")

    assert offloaded is True
    model.offload_to_cpu.assert_called_once_with()
    model.unload.assert_not_called()
    assert manager._object_reconstruction_model is model
    assert manager.get_object_reconstruction_model() is model
    release_cuda_memory.assert_called_once_with()


def test_released_model_is_lazily_created_again(monkeypatch):
    created = []

    class FakeCompletionAdapter:
        def __init__(self, config, device):
            created.append(self)

        def unload(self):
            pass

    fake_config = Mock()
    fake_config.device = "cuda"
    fake_config.get_model_config.return_value = {"name": "sdamodal"}
    monkeypatch.setattr(manager_module, "config", fake_config)
    monkeypatch.setattr(
        manager_module.ModelRegistry,
        "get_class",
        Mock(return_value=FakeCompletionAdapter),
    )
    monkeypatch.setattr(
        manager_module,
        "_release_cuda_memory",
        Mock(),
        raising=False,
    )
    manager = _new_manager()

    first = manager.get_completion_model()
    manager.release_model("completion")
    second = manager.get_completion_model()

    assert first is not second
    assert created == [first, second]


def test_release_missing_model_still_flushes_failed_load_allocations(
    monkeypatch,
):
    manager = _new_manager()
    release_cuda_memory = Mock()
    monkeypatch.setattr(
        manager_module,
        "_release_cuda_memory",
        release_cuda_memory,
    )

    released = manager.release_model("completion")

    assert released is False
    release_cuda_memory.assert_called_once_with()


def test_warmup_all_keeps_only_first_pipeline_stage_resident(monkeypatch):
    manager = _new_manager()
    segmentation_getter = Mock()
    matting_getter = Mock()
    background_getter = Mock()
    monkeypatch.setattr(
        manager, "get_segmentation_model", segmentation_getter
    )
    monkeypatch.setattr(manager, "get_matting_model", matting_getter)
    monkeypatch.setattr(
        manager,
        "get_background_inpainting_model",
        background_getter,
    )

    manager.warmup_all()

    segmentation_getter.assert_called_once_with()
    matting_getter.assert_not_called()
    background_getter.assert_not_called()


def test_background_inpainting_model_uses_its_own_category(monkeypatch):
    created = []

    class FakeBackgroundInpainter:
        def __init__(self, config, device):
            created.append((config, device))

    fake_config = Mock()
    fake_config.device = "cpu"
    fake_config.get_model_config.return_value = {
        "name": "lama",
        "feather_radius": 2.0,
    }
    monkeypatch.setattr(manager_module, "config", fake_config)
    get_class = Mock(return_value=FakeBackgroundInpainter)
    monkeypatch.setattr(manager_module.ModelRegistry, "get_class", get_class)

    manager = _new_manager()
    first = manager.get_background_inpainting_model()
    second = manager.get_background_inpainting_model()

    assert first is second
    assert created == [({"feather_radius": 2.0}, "cpu")]
    get_class.assert_called_once_with("background_inpainting", "lama")


def test_missing_object_reconstruction_model_is_reported_without_loading(
    monkeypatch,
):
    fake_config = Mock()
    fake_config.has_active_model.return_value = False
    monkeypatch.setattr(manager_module, "config", fake_config)
    get_class = Mock()
    monkeypatch.setattr(manager_module.ModelRegistry, "get_class", get_class)

    manager = _new_manager()

    assert manager.has_object_reconstruction_model() is False
    assert manager.get_object_reconstruction_model() is None
    get_class.assert_not_called()


def test_configured_object_reconstruction_uses_only_its_own_category(
    monkeypatch,
):
    class FakeObjectReconstructor:
        def __init__(self, config, device):
            self.config = config
            self.device = device

    fake_config = Mock()
    fake_config.device = "cuda"
    fake_config.has_active_model.return_value = True
    fake_config.get_model_config.return_value = {
        "name": "future_adapter",
        "input_size": 512,
    }
    monkeypatch.setattr(manager_module, "config", fake_config)
    get_class = Mock(return_value=FakeObjectReconstructor)
    monkeypatch.setattr(manager_module.ModelRegistry, "get_class", get_class)

    manager = _new_manager()
    model = manager.get_object_reconstruction_model()

    assert model.config == {"input_size": 512}
    assert model.device == "cuda"
    get_class.assert_called_once_with(
        "object_reconstruction", "future_adapter"
    )


def test_inpainting_configuration_has_two_isolated_categories():
    from backend.config import config as runtime_config

    config_path = Path(__file__).resolve().parents[1] / "backend" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    models = config["models"]
    assert "inpainting" not in models
    assert models["object_reconstruction"]["active"] == "hd_painter"
    assert "hd_painter" in models["object_reconstruction"]
    assert models["background_inpainting"]["active"] == "original_lama"
    assert {"lama", "original_lama", "sdxl"} <= set(
        models["background_inpainting"]
    )
    assert runtime_config.has_active_model("object_reconstruction") is True
    assert runtime_config.has_active_model("background_inpainting") is True


def test_completion_configuration_contains_initial_adapter_settings():
    config_path = Path(__file__).resolve().parents[1] / "backend" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    completion = config["models"]["completion"]

    assert completion["active"] == "sdamodal"
    assert completion["sdamodal"] == {
        "config_path": "backend/models/completion/config_SDAmodal.yaml",
        "checkpoint_path": "checkpoints/ckpt_SDAmodal.pth",
        "input_size": 512,
        "enlarge_box": 3.0,
        "lazy_load": True,
    }
