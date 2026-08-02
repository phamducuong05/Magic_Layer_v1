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
    "backend/models/background_inpainting/simple_lama/adapter.py",
    "backend/models/background_inpainting/original_lama/adapter.py",
    "backend/models/background_inpainting/original_lama/runtime.py",
    "backend/models/background_inpainting/sdxl/adapter.py",
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

    call = basic_config.call_args
    assert call.kwargs["level"] == "DEBUG"
    assert call.kwargs["force"] is True
    assert len(call.kwargs["handlers"]) == 1
    assert isinstance(
        call.kwargs["handlers"][0].formatter,
        logging_config.ColorFormatter,
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


def test_logging_configuration_is_available_from_yaml():
    from backend.config import config

    assert config.get_logging_config() == {
        "level": "INFO",
        "workflow_detail": "compact",
        "third_party_level": "WARNING",
    }


def test_log_event_defaults_to_debug_and_formats_readable_metadata(caplog):
    logging_config = _logging_module()
    logger = logging_config.get_logger("test.pipeline.trace")

    with caplog.at_level(logging.DEBUG, logger="test.pipeline.trace"):
        logging_config.log_event(
            logger,
            "completion",
            "decision",
            object_id="object-0",
            decision="accepted",
        )

    assert "[COMPLETION] DECISION" in caplog.text
    assert "decision=accepted" in caplog.text
    assert "object_id=object-0" in caplog.text
    assert "{" not in caplog.text


def test_log_event_can_promote_notable_decision_to_info(caplog):
    logging_config = _logging_module()
    logger = logging_config.get_logger("test.pipeline.summary")

    with caplog.at_level(logging.INFO, logger="test.pipeline.summary"):
        logging_config.log_event(
            logger,
            "reconstruction",
            "skip",
            level=logging.INFO,
            object_id="object-4",
            reason="no_directional_overlap",
        )

    assert "[RECONSTRUCTION] SKIP" in caplog.text
    assert "object_id=object-4" in caplog.text


def test_trace_stage_logs_start_complete_and_failure(caplog):
    logging_config = _logging_module()
    logger = logging_config.get_logger("test.pipeline.stage")

    with caplog.at_level(logging.DEBUG, logger="test.pipeline.stage"):
        with logging_config.trace_stage(logger, "matting", groups=2):
            pass
        with pytest.raises(RuntimeError, match="synthetic"):
            with logging_config.trace_stage(logger, "reconstruction"):
                raise RuntimeError("synthetic")

    assert "[MATTING] START" in caplog.text
    assert "[MATTING] COMPLETE" in caplog.text
    assert "duration_ms=" in caplog.text
    assert "[RECONSTRUCTION] FAILED" in caplog.text
    assert "error_type=RuntimeError" in caplog.text


def test_color_formatter_adds_ansi_without_mutating_log_message():
    logging_config = _logging_module()
    formatter = logging_config.ColorFormatter(
        "%(levelname)s %(message)s", use_colors=True
    )
    record = logging.LogRecord(
        "test",
        logging.INFO,
        __file__,
        1,
        "Finding components",
        (),
        None,
    )

    rendered = formatter.format(record)

    assert "\x1b[" in rendered
    assert "Finding components" in rendered
    assert record.msg == "Finding components"


def test_workflow_event_is_info_and_human_readable(caplog):
    logging_config = _logging_module()
    logger = logging_config.get_logger("test.workflow")

    with caplog.at_level(logging.INFO, logger="test.workflow"):
        logging_config.workflow_event(
            logger,
            "keywords",
            "Extracted keywords",
            keywords=["dog", "wooden chair"],
        )

    assert "[KEYWORDS] Extracted keywords" in caplog.text
    assert "keywords=dog,'wooden chair'" in caplog.text


def test_compact_filter_keeps_workflow_info_and_all_errors():
    logging_config = _logging_module()
    compact_filter = logging_config.CompactWorkflowFilter()
    ordinary = logging.LogRecord(
        "pipeline", logging.INFO, __file__, 1, "model detail", (), None
    )
    milestone = logging.LogRecord(
        "pipeline", logging.INFO, __file__, 1, "workflow", (), None
    )
    milestone.workflow = True
    failure = logging.LogRecord(
        "pipeline", logging.ERROR, __file__, 1, "failed", (), None
    )

    assert compact_filter.filter(ordinary) is False
    assert compact_filter.filter(milestone) is True
    assert compact_filter.filter(failure) is True


def test_log_metadata_truncates_very_long_values(caplog):
    logging_config = _logging_module()
    logger = logging_config.get_logger("test.workflow.error")

    with caplog.at_level(logging.ERROR, logger="test.workflow.error"):
        logging_config.log_event(
            logger,
            "pipeline",
            "failed",
            level=logging.ERROR,
            error="x" * 400,
        )

    assert "x" * 150 in caplog.text
    assert "x" * 200 not in caplog.text
    assert "..." in caplog.text
