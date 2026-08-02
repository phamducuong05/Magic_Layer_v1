# Logging and Processing Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add concise colored backend workflow logs and real polling-based frontend progress while preserving the existing synchronous API.

**Architecture:** A thread-safe in-memory job store owns job state, FastAPI schedules job processing, and the pipeline reports checkpoints through an optional callback. The frontend starts a job, polls its resource, and renders progress until it receives the existing process result.

**Tech Stack:** Python 3, FastAPI, stdlib logging/threading/asyncio, React 18, Vitest, Testing Library.

## Global Constraints

- Preserve `POST /api/process-image` behavior and response shape.
- Job progress is monotonic from 0 to 100 and represents real backend checkpoints.
- INFO logs contain only important workflow events; detailed diagnostics remain DEBUG.
- All user-facing frontend text is English.
- In-memory jobs support one backend process only.

---

### Task 1: Job state store

**Files:**
- Create: `backend/services/process_jobs.py`
- Create: `tests/test_process_jobs.py`

**Interfaces:**
- Produces: `ProcessJobStore.create()`, `update()`, `complete()`, `fail()`, and `get()` returning immutable serializable snapshots.

- [ ] Write tests proving initial state, monotonic progress, completed result, failed error, and missing-ID behavior.
- [ ] Run `pytest tests/test_process_jobs.py -q` and verify failure because the module is missing.
- [ ] Implement the minimal lock-protected store with copied snapshots.
- [ ] Run the focused tests and verify they pass.

### Task 2: Pipeline progress and compact colored logs

**Files:**
- Modify: `backend/core/logging.py`
- Modify: `backend/pipeline/orchestrator.py`
- Modify: `backend/main.py`
- Modify: `tests/test_logging_config.py`
- Modify: `tests/test_pipeline_trace_logging.py`
- Create: `tests/test_pipeline_progress.py`

**Interfaces:**
- Consumes: callback `(stage: str, progress: int, message: str) -> None`.
- Produces: `workflow_event()` and optional `progress_callback` on `process_image()`.

- [ ] Write failing tests for ANSI formatting, compact INFO behavior, callback checkpoints, overlap/reconstruction summary, and final group count.
- [ ] Run the focused backend tests and confirm the expected failures.
- [ ] Implement the formatter, promote only workflow milestones to INFO, and emit pipeline callbacks.
- [ ] Re-run the focused tests and verify they pass.

### Task 3: Background job endpoints

**Files:**
- Modify: `backend/main.py`
- Modify: `tests/test_main_claude_keywords.py`
- Create: `tests/test_process_job_api.py`

**Interfaces:**
- Produces: `POST /api/process-image/jobs` and `GET /api/process-image/jobs/{job_id}`.

- [ ] Write failing route tests for accepted, processing, completed, failed, invalid upload, and unknown job responses.
- [ ] Run the route tests and verify failure because the endpoints are missing.
- [ ] Extract shared upload/response helpers, schedule keyword resolution plus `asyncio.to_thread(process_image)`, and update the store through the progress callback.
- [ ] Run all related backend tests and verify they pass.

### Task 4: Frontend polling client

**Files:**
- Modify: `frontend/src/api.js`
- Modify: `frontend/src/api.test.js`

**Interfaces:**
- Produces: `processImage(file, keywords, onProgress, fetchImpl)` resolving to the existing result shape.

- [ ] Write failing tests for job creation, polling progress, completion, and failure.
- [ ] Run `npm test -- src/api.test.js` and confirm expected failures.
- [ ] Implement bounded polling with injectable fetch and delay dependencies.
- [ ] Re-run the API tests and verify they pass.

### Task 5: Frontend progress panel

**Files:**
- Create: `frontend/src/components/ProcessingProgress.jsx`
- Modify: `frontend/src/components/UploadScreen.jsx`
- Modify: `frontend/src/components/UploadScreen.test.jsx`
- Modify: `frontend/src/index.css`

**Interfaces:**
- Consumes: `{ stage, progress, message }` updates from `processImage`.

- [ ] Write failing UI tests showing live stage/message/percentage and successful handoff to the workspace.
- [ ] Run the focused component tests and confirm expected failures.
- [ ] Implement the accessible progress panel and connect the callback without changing upload validation.
- [ ] Run the focused frontend tests and verify they pass.

### Task 6: Full verification

**Files:**
- Modify only files required by failures caused by this feature.

- [ ] Run focused and full backend tests.
- [ ] Run `npm test`, `npm run lint`, and `npm run build` in `frontend`.
- [ ] Review `git diff --check` and `git status --short`.
- [ ] Commit the implementation with a concise feature message.
