"""Thread-safe in-memory state for background image-processing jobs."""

from copy import deepcopy
from dataclasses import dataclass, replace
from threading import RLock
from typing import Any, Optional
from uuid import uuid4


@dataclass(frozen=True)
class ProcessJobSnapshot:
    job_id: str
    status: str
    stage: str
    progress: int
    message: str
    result: Optional[Any] = None
    error: Optional[str] = None


class ProcessJobStore:
    """Own process-local job state and return isolated snapshots."""

    def __init__(self, *, max_jobs: int = 20) -> None:
        if max_jobs < 1:
            raise ValueError("max_jobs must be positive")
        self._jobs: dict[str, ProcessJobSnapshot] = {}
        self._max_jobs = max_jobs
        self._lock = RLock()

    def create(self) -> ProcessJobSnapshot:
        snapshot = ProcessJobSnapshot(
            job_id=uuid4().hex,
            status="queued",
            stage="queued",
            progress=0,
            message="Waiting to start",
        )
        with self._lock:
            self._evict_terminal_jobs()
            self._jobs[snapshot.job_id] = snapshot
        return self._copy(snapshot)

    def update(
        self,
        job_id: str,
        stage: str,
        progress: int,
        message: str,
    ) -> ProcessJobSnapshot:
        with self._lock:
            current = self._require(job_id)
            if current.status in {"completed", "failed"}:
                return self._copy(current)
            normalized_progress = min(99, max(0, int(progress)))
            if normalized_progress < current.progress:
                return self._copy(current)
            updated = replace(
                current,
                status="processing",
                stage=stage,
                progress=normalized_progress,
                message=message,
            )
            self._jobs[job_id] = updated
            return self._copy(updated)

    def complete(self, job_id: str, result: Any) -> ProcessJobSnapshot:
        with self._lock:
            current = self._require(job_id)
            updated = replace(
                current,
                status="completed",
                stage="completed",
                progress=100,
                message="Layers are ready",
                result=deepcopy(result),
                error=None,
            )
            self._jobs[job_id] = updated
            return self._copy(updated)

    def fail(self, job_id: str, error: str) -> ProcessJobSnapshot:
        with self._lock:
            current = self._require(job_id)
            updated = replace(
                current,
                status="failed",
                stage="failed",
                message="Processing failed",
                result=None,
                error=error,
            )
            self._jobs[job_id] = updated
            return self._copy(updated)

    def get(self, job_id: str) -> ProcessJobSnapshot:
        with self._lock:
            return self._copy(self._require(job_id))

    def _require(self, job_id: str) -> ProcessJobSnapshot:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise KeyError(job_id) from None

    def _evict_terminal_jobs(self) -> None:
        for existing_id, snapshot in tuple(self._jobs.items()):
            if len(self._jobs) < self._max_jobs:
                break
            if snapshot.status in {"completed", "failed"}:
                del self._jobs[existing_id]

    @staticmethod
    def _copy(snapshot: ProcessJobSnapshot) -> ProcessJobSnapshot:
        return replace(
            snapshot,
            result=deepcopy(snapshot.result),
        )
