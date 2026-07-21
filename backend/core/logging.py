"""Central logging configuration for application-owned backend modules."""

from collections.abc import Iterator
from contextlib import contextmanager
import json
import logging as stdlib_logging
from time import perf_counter
from typing import Any, Union


DEFAULT_LOG_FORMAT = (
    "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
CLI_LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"
LogLevel = Union[int, str]


def configure_logging(
    *,
    level: LogLevel = stdlib_logging.INFO,
    log_format: str = DEFAULT_LOG_FORMAT,
    force: bool = False,
) -> None:
    """Configure root logging for one application entry point."""
    stdlib_logging.basicConfig(
        level=level,
        format=log_format,
        force=force,
    )


def get_logger(name: str) -> stdlib_logging.Logger:
    """Return a logger that preserves the caller's module name."""
    return stdlib_logging.getLogger(name)


def log_event(
    logger: stdlib_logging.Logger,
    stage: str,
    event: str,
    **metadata: Any,
) -> None:
    """Emit one INFO pipeline event with deterministic JSON metadata."""
    logger.info(
        "[%s] %s %s",
        stage.upper(),
        event.upper(),
        json.dumps(metadata, sort_keys=True, default=str),
    )


@contextmanager
def trace_stage(
    logger: stdlib_logging.Logger,
    stage: str,
    **metadata: Any,
) -> Iterator[None]:
    """Log the start, duration, and failure of one pipeline stage."""
    started_at = perf_counter()
    log_event(logger, stage, "start", **metadata)
    try:
        yield
    except Exception as exc:
        log_event(
            logger,
            stage,
            "failed",
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
            duration_ms=round((perf_counter() - started_at) * 1000.0, 3),
        )
