# Manual Layer Occluder Prompt Design

## Goal

Update only the manual layer-extraction prompt so it reuses the same
`COMMON_STRICT_RULES` as automated extraction while limiting its additional
task to finding real occluders of user-selected targets.

## Scope

- Change only `OCCLUDER_PROMPT` in `backend/services/claude_vision.py`.
- Add or update prompt-focused tests in `tests/test_claude_vision.py`.
- Do not change `FOREGROUND_OBJECT_PROMPT`, automated extraction behavior,
  schemas, result parsing, ranking, quotas, or pipeline code.

## Manual Prompt Behavior

For each user-supplied target keyword, the model must:

1. Inspect the entire image and identify every visible instance matched by the
   target keyword, including instances near corners and frame boundaries.
2. Evaluate occlusion separately for every matched instance.
3. Return the union of independent objects that are closer to the camera and
   physically hide at least one visible part of at least one matched instance.
4. Exclude objects that are merely nearby, touching, intersecting a bounding
   box or silhouette, or visually overlapping without actually hiding the
   target.
5. Return an empty occluder list only when no real occluder exists for any
   matched instance.

The existing target-refinement rule remains in place. The target itself must
not be returned as its own occluder. Existing whole-object and parent-object
rules from `COMMON_STRICT_RULES` remain authoritative.

## Verification

Prompt tests will assert that manual mode explicitly requires:

- all instances of every target keyword;
- real physical hiding and foreground depth;
- unioning occluders across instances;
- rejection of proximity, touching, and bounding-box-only overlap.

A regression assertion will capture the automated prompt before the change and
confirm it remains byte-for-byte unchanged after the manual prompt edit.
