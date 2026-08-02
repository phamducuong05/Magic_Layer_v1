"""Central logging configuration for application-owned backend modules."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import logging as stdlib_logging
import os
from time import perf_counter
from typing import Any, Optional, Union


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
CLI_LOG_FORMAT = "%(levelname)s | %(message)s"
LogLevel = Union[int, str]
THIRD_PARTY_LOGGERS = (
    "diffusers",
    "lightning_fabric",
    "open_clip",
    "PIL",
    "torch",
    "transformers",
)


class ColorFormatter(stdlib_logging.Formatter):
    """Add readable terminal colors without changing the log record."""

    COLORS = {
        stdlib_logging.DEBUG: "\033[90m",
        stdlib_logging.INFO: "\033[96m",
        stdlib_logging.WARNING: "\033[93m",
        stdlib_logging.ERROR: "\033[91m",
        stdlib_logging.CRITICAL: "\033[95m",
    }
    RESET = "\033[0m"

    def __init__(
        self,
        fmt: str,
        *,
        use_colors: Optional[bool] = None,
    ) -> None:
        super().__init__(fmt)
        self.use_colors = (
            not bool(os.environ.get("NO_COLOR"))
            if use_colors is None
            else use_colors
        )

    def format(self, record: stdlib_logging.LogRecord) -> str:
        rendered = super().format(record)
        if not self.use_colors:
            return rendered
        color = self.COLORS.get(record.levelno, "")
        return f"{color}{rendered}{self.RESET}" if color else rendered


class CompactWorkflowFilter(stdlib_logging.Filter):
    """Hide noisy application INFO records while keeping warnings and milestones."""

    def filter(self, record: stdlib_logging.LogRecord) -> bool:
        if record.levelno != stdlib_logging.INFO:
            return True
        return bool(getattr(record, "workflow", False))


def configure_logging(
    *,
    level: LogLevel = stdlib_logging.INFO,
    log_format: str = DEFAULT_LOG_FORMAT,
    force: bool = False,
    third_party_level: LogLevel = stdlib_logging.WARNING,
) -> None:
    """Configure root logging for one application entry point."""
    handler = stdlib_logging.StreamHandler()
    handler.setFormatter(ColorFormatter(log_format))
    handler.addFilter(CompactWorkflowFilter())
    stdlib_logging.basicConfig(
        level=level,
        handlers=[handler],
        force=force,
    )
    for logger_name in THIRD_PARTY_LOGGERS:
        stdlib_logging.getLogger(logger_name).setLevel(third_party_level)


def get_logger(name: str) -> stdlib_logging.Logger:
    """Return a logger that preserves the caller's module name."""
    return stdlib_logging.getLogger(name)


def log_event(
    logger: stdlib_logging.Logger,
    stage: str,
    event: str,
    *,
    level: LogLevel = stdlib_logging.DEBUG,
    **metadata: Any,
) -> None:
    """Emit one readable pipeline event at the requested verbosity."""
    suffix = _format_metadata(metadata)
    logger.log(
        _coerce_level(level),
        "[%s] %s%s",
        stage.upper(),
        event.upper(),
        f" {suffix}" if suffix else "",
    )


def workflow_event(
    logger: stdlib_logging.Logger,
    stage: str,
    message: str,
    **metadata: Any,
) -> None:
    """Emit one concise INFO milestone intended for the server console."""
    suffix = _format_metadata(metadata)
    logger.info(
        "[%s] %s%s",
        stage.upper(),
        message,
        f" | {suffix}" if suffix else "",
        extra={"workflow": True},
    )


def _coerce_level(level: LogLevel) -> int:
    if isinstance(level, int):
        return level
    resolved = stdlib_logging.getLevelName(level.upper())
    if not isinstance(resolved, int):
        raise ValueError(f"unsupported log level: {level!r}")
    return resolved


def _format_value(value: Any) -> str:
    """Format metadata compactly without multi-line JSON payloads."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, Mapping):
        return ",".join(
            f"{key}:{_format_value(item)}"
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return ",".join(_format_value(item) for item in value)
    rendered = str(value)
    if len(rendered) > 160:
        rendered = f"{rendered[:157]}..."
    return repr(rendered) if any(char.isspace() for char in rendered) else rendered


def _format_metadata(metadata: Mapping[str, Any]) -> str:
    return " ".join(
        f"{key}={_format_value(value)}" for key, value in sorted(metadata.items())
    )


@contextmanager
def trace_stage(
    logger: stdlib_logging.Logger,
    stage: str,
    **metadata: Any,
) -> Iterator[None]:
    """Log the start, duration, and failure of one pipeline stage."""
    started_at = perf_counter()
    log_event(logger, stage, "start", level=stdlib_logging.DEBUG, **metadata)
    try:
        yield
    except Exception as exc:
        log_event(
            logger,
            stage,
            "failed",
            level=stdlib_logging.ERROR,
            duration_ms=round((perf_counter() - started_at) * 1000.0, 3),
            error=str(exc) or type(exc).__name__,
            error_type=type(exc).__name__,
        )
        raise
    else:
        log_event(
            logger,
            stage,
            "complete",
            level=stdlib_logging.DEBUG,
            duration_ms=round((perf_counter() - started_at) * 1000.0, 3),
        )
