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
    "inpainting": ["lama", "sdxl"],
}.items():
    package_name = f"backend.models.{category}"
    package = types.ModuleType(package_name)
    package.__path__ = []
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
    monkeypatch.setattr(manager, "get_inpainting_model", Mock())
    completion_getter = Mock()
    monkeypatch.setattr(
        manager,
        "get_completion_model",
        completion_getter,
        raising=False,
    )

    manager.warmup_all()

    completion_getter.assert_not_called()


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
