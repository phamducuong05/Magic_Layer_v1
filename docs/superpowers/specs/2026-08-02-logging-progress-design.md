# Logging and Processing Progress Design

## Goal

Provide concise, colored backend workflow logs and real frontend progress while an image is processed, without changing the existing synchronous `/api/process-image` contract.

## Architecture

The pipeline accepts an optional progress callback. It emits a small fixed set of user-facing checkpoints while existing detailed diagnostic events remain available at `DEBUG`. A dedicated workflow log helper renders important `INFO` events with ANSI color and compact metadata.

The backend adds an in-memory, thread-safe job store and two endpoints:

- `POST /api/process-image/jobs` validates and copies the upload, creates a job, and returns HTTP 202 immediately.
- `GET /api/process-image/jobs/{job_id}` returns the latest status and, when complete, the existing process response.

Each job runs keyword extraction asynchronously and the blocking model pipeline in a worker thread. The existing synchronous endpoint remains unchanged for compatibility. The in-memory store is intentionally scoped to the current single-process model server; a multi-worker deployment must replace it with a shared store such as Redis.

The store retains at most 20 jobs under normal operation and evicts the oldest terminal jobs first. Active jobs are never evicted, even when the temporary limit is exceeded.

## Progress Contract

Job status contains `job_id`, `status`, `stage`, `progress`, and `message`. `status` is one of `queued`, `processing`, `completed`, or `failed`; `progress` is an integer from 0 through 100. Completed jobs include `result`; failed jobs include a stable English `error`.

The main checkpoints are keyword analysis, segmentation, overlap analysis, reconstruction, grouping, layer extraction, background cleanup, and completion. Progress never decreases. The frontend polls until a terminal state and shows the current message, percentage, progress bar, and checkpoint list.

## Logging

Console output at `INFO` contains only workflow milestones:

- extracted keywords;
- current important stage;
- overlap/reconstruction pairs when present;
- final group count;
- completion or failure.

Detailed per-object, per-pair, and stage timing logs remain available at `DEBUG`. ANSI colors are added by a formatter, so log record messages stay clean for tests and structured consumers. `NO_COLOR` disables color.

## Error Handling

Upload and keyword validation use the same rules as the synchronous endpoint. Background failures are captured in the job instead of escaping the task. Unknown or expired job IDs return 404. The frontend stops polling on failure, displays a stable English error, and allows another submission.

## Testing

Backend tests cover monotonic job updates, terminal states, job API responses, progress callbacks, compact INFO logging, and ANSI formatting. Frontend tests cover job creation/polling, progress callbacks, progress UI updates, successful transition to the canvas, and failures. Existing backend and frontend suites, lint, and production build must remain green.
