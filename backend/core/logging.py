"""Central logging configuration for application-owned backend modules."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import logging as stdlib_logging
from time import perf_counter
from typing import Any, Union


DEFAULT_LOG_FORMAT = (
    "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
CLI_LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"
LogLevel = Union[int, str]
THIRD_PARTY_LOGGERS = (
    "diffusers",
    "lightning_fabric",
    "open_clip",
    "PIL",
    "torch",
    "transformers",
)


def configure_logging(
    *,
    level: LogLevel = stdlib_logging.INFO,
    log_format: str = DEFAULT_LOG_FORMAT,
    force: bool = False,
    third_party_level: LogLevel = stdlib_logging.WARNING,
) -> None:
    """Configure root logging for one application entry point."""
    stdlib_logging.basicConfig(
        level=level,
        format=log_format,
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
    log_event(logger, stage, "start", level=stdlib_logging.INFO, **metadata)
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
            level=stdlib_logging.INFO,
            duration_ms=round((perf_counter() - started_at) * 1000.0, 3),
        )
