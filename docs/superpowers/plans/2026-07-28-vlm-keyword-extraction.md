# VLM Keyword Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate validated English SAM3 keywords from an uploaded image with Claude when no manual override is supplied.

**Architecture:** A provider-neutral `KeywordExtractor` protocol owns the async extraction contract. `ClaudeVisionKeywordExtractor` sends an in-memory resized image to Anthropic with a JSON schema, while the FastAPI boundary chooses manual override or VLM output before calling the unchanged image pipeline.

**Tech Stack:** Python, FastAPI, Pillow, Pydantic, Anthropic Python SDK, pytest

## Global Constraints

- Preserve `process_image(image, Sequence[str])`.
- Never store or log the Anthropic API key or image base64.
- Use `AsyncAnthropic`; no synchronous network call in the async endpoint.
- Keep manual keywords only as a transitional override.
- Return 1-10 normalized keywords.

---

### Task 1: Keyword domain contract and configuration

**Files:**
- Create: `backend/services/__init__.py`
- Create: `backend/services/keyword_extractor.py`
- Modify: `backend/config.py`
- Modify: `backend/config.yaml`
- Test: `tests/test_keyword_extractor.py`

**Interfaces:**
- Produces: `KeywordExtractor.extract_keywords(image) -> list[str]`
- Produces: `normalize_keywords(values, max_keywords, max_length) -> list[str]`
- Produces: `ConfigManager.get_vlm_config() -> dict[str, Any]`

- [x] Write tests that reject empty/too-long/too-many values and deduplicate case-insensitively.
- [x] Run the tests and verify failure because the service module does not exist.
- [x] Implement the protocol, domain errors, normalization, and VLM config validation.
- [x] Run the tests and verify they pass.

### Task 2: Claude structured-output adapter

**Files:**
- Create: `backend/services/claude_vision.py`
- Test: `tests/test_claude_vision.py`

**Interfaces:**
- Consumes: the domain contract and merged VLM settings from Task 1.
- Produces: `ClaudeVisionKeywordExtractor(settings, client=None)`.

- [x] Write async tests with a fake client for the exact image/text payload, JSON schema parsing, missing key, refusal, truncation, malformed output, and transient upstream errors.
- [x] Run the tests and verify failure because the adapter does not exist.
- [x] Implement in-memory JPEG encoding, lazy `AsyncAnthropic`, `messages.create`, stop-reason checks, JSON parsing, and error translation.
- [x] Run the tests and verify they pass.

### Task 3: FastAPI routing

**Files:**
- Modify: `backend/main.py`
- Create: `tests/test_main_claude_keywords.py`

**Interfaces:**
- Consumes: `KeywordExtractor`.
- Produces: `resolve_keywords(image, supplied_keywords, extractor) -> list[str]`.

- [x] Write async boundary tests proving manual keywords skip VLM, omitted keywords use VLM, and domain failures map to 502/503.
- [x] Run the tests and verify failure because auto resolution is absent.
- [x] Make the form field optional, add a replaceable extractor provider, resolve keywords, and map safe HTTP errors.
- [x] Run the tests and verify they pass.

### Task 4: Runtime packaging and regression verification

**Files:**
- Modify: `backend/requirements.txt`
- Create: `.env.example`

- [x] Add `anthropic>=0.104,<1` and the environment placeholder.
- [x] Run the three new test modules.
- [x] Run the full pytest suite.
- [x] Review the diff for secrets, blocking calls, raw image logging, and scope drift.
