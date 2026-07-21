"""Tests for centralized application logging configuration."""

import importlib
import logging
from pathlib import Path
from unittest.mock import Mock

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPLICATION_LOGGING_FILES = [
    "backend/main.py",
    "backend/run_mask_completion.py",
    "backend/models/manager.py",
    "backend/models/segmentation/sam3.py",
    "backend/models/matting/birefnet.py",
    "backend/models/background_inpainting/lama.py",
    "backend/models/background_inpainting/sdxl.py",
    "backend/models/object_reconstruction/adapter.py",
    "backend/pipeline/diagnostics.py",
    "backend/pipeline/orchestrator.py",
    "backend/pipeline/layers.py",
    "backend/pipeline/segmentation.py",
    "backend/pipeline/completion/completion.py",
    "backend/pipeline/completion/validate_completion.py",
    "backend/pipeline/reconstruction/reconstruction.py",
]


def _logging_module():
    try:
        return importlib.import_module("backend.core.logging")
    except ModuleNotFoundError:
        pytest.fail("backend.core.logging has not been implemented")


def test_configure_logging_owns_the_application_format(monkeypatch):
    logging_config = _logging_module()
    basic_config = Mock()
    monkeypatch.setattr(logging_config.stdlib_logging, "basicConfig", basic_config)

    logging_config.configure_logging(level="DEBUG", force=True)

    basic_config.assert_called_once_with(
        level="DEBUG",
        format=logging_config.DEFAULT_LOG_FORMAT,
        force=True,
    )


def test_get_logger_preserves_calling_module_name():
    logging_config = _logging_module()

    logger = logging_config.get_logger("backend.pipeline.example")

    assert logger.name == "backend.pipeline.example"


def test_application_modules_use_central_logging_api():
    for relative_path in APPLICATION_LOGGING_FILES:
        source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        assert "import logging" not in source, relative_path
        assert "logging.basicConfig" not in source, relative_path
        assert "get_logger" in source, relative_path


def test_entry_points_configure_logging_centrally():
    for relative_path in ("backend/main.py", "backend/run_mask_completion.py"):
        source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        assert "configure_logging(" in source, relative_path


def test_log_event_emits_structured_info_metadata(caplog):
    logging_config = _logging_module()
    logger = logging_config.get_logger("test.pipeline.trace")

    with caplog.at_level(logging.INFO, logger="test.pipeline.trace"):
        logging_config.log_event(
            logger,
            "completion",
            "decision",
            object_id="object-0",
            decision="accepted",
        )

    assert '[COMPLETION] DECISION {' in caplog.text
    assert '"decision": "accepted"' in caplog.text
    assert '"object_id": "object-0"' in caplog.text


def test_trace_stage_logs_start_complete_and_failure(caplog):
    logging_config = _logging_module()
    logger = logging_config.get_logger("test.pipeline.stage")

    with caplog.at_level(logging.INFO, logger="test.pipeline.stage"):
        with logging_config.trace_stage(logger, "matting", groups=2):
            pass
        with pytest.raises(RuntimeError, match="synthetic"):
            with logging_config.trace_stage(logger, "reconstruction"):
                raise RuntimeError("synthetic")

    assert "[MATTING] START" in caplog.text
    assert "[MATTING] COMPLETE" in caplog.text
    assert '"duration_ms":' in caplog.text
    assert "[RECONSTRUCTION] FAILED" in caplog.text
    assert '"error_type": "RuntimeError"' in caplog.text
