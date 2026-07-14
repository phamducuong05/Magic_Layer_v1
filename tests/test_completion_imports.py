"""Regression tests for package-safe SDAmodal production imports."""

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = PROJECT_ROOT / "backend" / "models"


def test_completion_inference_imports_without_ambiguous_or_debug_packages(
    tmp_path,
):
    script = f"""
import importlib.abc
import pathlib
import sys
import types

blocked = {{"models", "utils", "inference", "src", "pdb", "ipdb", "matplotlib", "pycocotools"}}

class BlockedImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockedImports())
models_package = types.ModuleType("backend.models")
models_package.__path__ = [r"{MODELS_PATH}"]
sys.modules["backend.models"] = models_package

import backend.models.completion.inference
import backend.models.completion.models.aw_sdm
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_completion_package_has_explicit_initializer():
    assert (MODELS_PATH / "completion" / "__init__.py").is_file()
