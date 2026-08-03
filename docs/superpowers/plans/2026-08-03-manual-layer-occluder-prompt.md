# Manual Layer Occluder Prompt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make manual layer extraction return only real occluders across every visible instance of each user keyword without changing automated extraction.

**Architecture:** Keep the existing shared `COMMON_STRICT_RULES` composition. Tighten only `OCCLUDER_PROMPT`, using prompt-level regression tests to define exhaustive instance coverage and real physical occlusion while preserving `FOREGROUND_OBJECT_PROMPT` byte-for-byte.

**Tech Stack:** Python 3, pytest, Anthropic structured-output prompt adapter

## Global Constraints

- Change only `OCCLUDER_PROMPT` in `backend/services/claude_vision.py`.
- Add or update prompt-focused tests in `tests/test_claude_vision.py`.
- Do not change `FOREGROUND_OBJECT_PROMPT`, automated extraction behavior, schemas, result parsing, ranking, quotas, or pipeline code.
- Manual mode must inspect every visible instance of every supplied target keyword.
- An occluder must be closer to the camera and physically hide part of a matched target instance.

---

### Task 1: Define and implement exhaustive real-occluder semantics

**Files:**
- Modify: `tests/test_claude_vision.py`
- Modify: `backend/services/claude_vision.py`

**Interfaces:**
- Consumes: `ClaudeVisionKeywordExtractor.extract_keywords(image, target_keywords)` and the existing `COMMON_STRICT_RULES` string.
- Produces: revised `OCCLUDER_PROMPT: str`; no Python API or schema changes.

- [ ] **Step 1: Write failing prompt regression tests**

Add assertions to the manual-mode test after normalizing whitespace:

```python
manual_prompt = " ".join(request["system"].split())
assert "identify every visible instance matched by that target" in manual_prompt
assert "evaluate occlusion separately for every matched instance" in manual_prompt
assert "union of independent objects" in manual_prompt
assert "closer to the camera and physically hide" in manual_prompt
assert "bounding-box or silhouette overlap" in manual_prompt
```

Protect automated mode with an equality assertion against the unchanged exported constant:

```python
from backend.services.claude_vision import FOREGROUND_OBJECT_PROMPT

assert request["system"] == FOREGROUND_OBJECT_PROMPT
```

- [ ] **Step 2: Run the focused tests and verify the new manual assertions fail**

Run:

```powershell
python -m pytest tests/test_claude_vision.py -k "indexed_occluder_mode or automatic_prompt" -v
```

Expected: the new manual assertions fail because the current task text does not explicitly require every target instance or unioning occluders across instances; the automated equality assertion passes.

- [ ] **Step 3: Tighten only `OCCLUDER_PROMPT`**

Replace the current broad manual task with text equivalent to:

```python
"""For each target, inspect the entire image and identify every visible
instance matched by that target, including instances near corners and frame
boundaries. Evaluate occlusion separately for every matched instance. Return
the union of independent objects that are closer to the camera and physically
hide at least one visible part of at least one matched instance. Objects that
are merely nearby, touching, intersecting, or have bounding-box or silhouette
overlap without physically hiding the target are not occluders."""
```

Retain target refinement, empty-list behavior, target-self exclusion, ordering, parent-object rules, the `COMMON_STRICT_RULES` concatenation, and JSON-only output.

- [ ] **Step 4: Run focused and full adapter tests**

Run:

```powershell
python -m pytest tests/test_claude_vision.py -v
```

Expected: all tests in the file pass. If pre-existing assertions describe older prompt copy, update only those prompt-copy assertions to the current shared rules; do not change runtime behavior outside `OCCLUDER_PROMPT`.

- [ ] **Step 5: Verify the diff is manual-only**

Run:

```powershell
git diff --check
git diff -- backend/services/claude_vision.py tests/test_claude_vision.py
```

Expected: production changes are confined to `OCCLUDER_PROMPT`; `FOREGROUND_OBJECT_PROMPT`, schemas, parsing, ranking, quotas, and pipeline code have no diff.
