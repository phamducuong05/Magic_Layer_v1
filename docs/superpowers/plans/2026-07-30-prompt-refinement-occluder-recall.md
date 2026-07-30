# Prompt Refinement and Occluder Recall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve user target semantics while exhaustively collecting per-target occluder keywords without allowing targets to consume the occluder quota.

**Architecture:** Add a deterministic prompt-refinement policy module, then change the Claude manual-mode schema from global arrays to indexed per-target results. The Claude adapter validates target coverage, semantic refinements, person cardinality, and ranked occluder union before returning the existing flat provider-neutral result consumed by `main.py`.

**Tech Stack:** Python 3.10, dataclasses, Anthropic structured outputs, FastAPI, pytest, Conda environment `layer`.

## Global Constraints

- Work only on `feature/cross-class-containment-grouping`.
- Run every Python test with `C:\Users\admin\anaconda3\envs\layer\python.exe`.
- Preserve manual target object type, specificity, and singular/plural cardinality.
- A single-token manual target always passes through unchanged after trimming.
- Permit synonym replacement only through an auditable code-owned alias map.
- For one to three automatically detected people, reject `people`; allow `people` only above three.
- The people-cardinality rule never rewrites a manual target.
- Tiny confirmed occluders must bypass the normal tiny/incidental exclusion.
- Body parts, clothing, wearables, and accessories remain attached to their person parent.
- Keep independent limits of 10 targets and 10 occluders.
- Do not modify segmentation, grouping, completion, reconstruction, matting, layer extraction, or background inpainting.

---

### Task 1: Deterministic target-refinement policy

**Files:**
- Create: `backend/services/prompt_refinement.py`
- Create: `tests/test_prompt_refinement.py`

**Interfaces:**
- Produces: `validate_refined_target(source: str, candidate: str) -> str`
- Produces: `validate_people_keyword_usage(keywords: Sequence[str], visible_person_count: int, *, allow_manual_people: bool = False) -> None`
- Produces: `union_ranked_occluders(per_target: Sequence[Sequence[str]], *, target_keywords: Sequence[str], max_occluders: int) -> list[str]`
- Raises: `InvalidKeywordExtraction` for impossible person counts or forbidden automatic `people`

- [ ] **Step 1: Write failing semantic-preservation tests**

```python
def test_single_token_target_never_uses_model_generalization():
    assert validate_refined_target("man", "people") == "man"


def test_long_description_can_keep_a_known_head_noun():
    assert validate_refined_target("tall man in red", "man") == "man"


def test_controlled_alias_can_simplify_an_object_name():
    assert validate_refined_target(
        "red four-door passenger automobile", "car"
    ) == "car"


def test_number_changing_candidate_falls_back_to_source():
    assert validate_refined_target("two men", "man") == "two men"
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m pytest tests/test_prompt_refinement.py -q
```

Expected: collection fails because `backend.services.prompt_refinement` does not exist.

- [ ] **Step 3: Implement the minimal controlled vocabulary policy**

Create a focused module containing:

```python
SIMPLE_OBJECT_VOCAB = frozenset({
    "boy", "boys", "car", "cars", "chair", "chairs", "girl", "girls",
    "man", "men", "person", "persons", "table", "tables", "woman", "women",
})

SIMPLE_VOCAB_ALIASES = {
    "automobile": "car",
    "automobiles": "cars",
}


def validate_refined_target(source: str, candidate: str) -> str:
    normalized_source = " ".join(source.split())
    normalized_candidate = " ".join(candidate.split())
    if len(normalized_source.split()) == 1:
        return normalized_source
    if _candidate_is_source_head(normalized_source, normalized_candidate):
        return normalized_candidate
    if _candidate_is_controlled_alias(normalized_source, normalized_candidate):
        return normalized_candidate
    return normalized_source
```

The private head check must use whole case-folded tokens, require the candidate
to be in `SIMPLE_OBJECT_VOCAB`, and reject singular/plural changes by allowing
only exact source tokens. The alias check must require an exact whole source
token or phrase key whose configured value equals the candidate.

- [ ] **Step 4: Add people-cardinality and ranked-union tests**

```python
def test_small_visible_group_cannot_be_generalized_to_people():
    with pytest.raises(InvalidKeywordExtraction, match="people"):
        validate_people_keyword_usage(["people"], 3)


def test_people_is_allowed_above_three_visible_people():
    validate_people_keyword_usage(["people"], 4)


def test_ranked_union_gives_each_target_a_first_occluder_before_seconds():
    result = union_ranked_occluders(
        [["man", "tree"], ["Man", "chair"]],
        target_keywords=["car", "table"],
        max_occluders=10,
    )
    assert result == ["man", "chair", "tree"]
```

- [ ] **Step 5: Implement cardinality validation and ranked union**

`visible_person_count` must be a non-negative integer and not a `bool`.
`union_ranked_occluders()` must iterate occluder rank first and target index
second, remove case-insensitive duplicates, remove target duplicates, preserve
the first spelling, and stop only at `max_occluders`.

- [ ] **Step 6: Run Task 1 tests and verify GREEN**

Run:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m pytest tests/test_prompt_refinement.py tests/test_keyword_extractor.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 1**

```powershell
git add backend/services/prompt_refinement.py tests/test_prompt_refinement.py
git commit -m "feat: validate prompt refinements"
```

---

### Task 2: Per-target Claude structured output and exhaustive occluder prompt

**Files:**
- Modify: `backend/services/keyword_extractor.py`
- Modify: `backend/services/claude_vision.py`
- Modify: `backend/services/__init__.py`
- Modify: `tests/test_claude_vision.py`

**Interfaces:**
- Consumes: `validate_refined_target`, `validate_people_keyword_usage`, and `union_ranked_occluders` from Task 1
- Produces: `TargetKeywordExtraction(input_index: int, source_keyword: str, keyword: str, occluders: tuple[str, ...])`
- Extends: `KeywordExtractionResult.target_results: tuple[TargetKeywordExtraction, ...]`
- Extends: `KeywordExtractionResult.visible_person_count: int | None`
- Preserves: `KeywordExtractionResult.keywords` and `.occluders`

- [ ] **Step 1: Write failing schema and prompt-contract tests**

Update manual fake responses to:

```json
{
  "visible_person_count": 1,
  "target_results": [
    {
      "input_index": 0,
      "refined_keyword": "car",
      "occluders": ["man", "tree"]
    }
  ]
}
```

Assert that manual schema requires `visible_person_count` and
`target_results`, and each item requires `input_index`, `refined_keyword`, and
`occluders`. Assert both system prompts say that a confirmed occluder must be
returned regardless of tiny size. Assert the manual prompt says every target
must be inspected and `hand`, `glasses`, `hat`, and clothing resolve to the
person parent.

- [ ] **Step 2: Write failing parsing and validation tests**

Add tests proving:

```python
assert result.keywords == ["man"]  # model proposed "people" for manual man
assert result.occluders == ["chair"]
assert result.target_results[0].input_index == 0
```

Also add parameterized failures for missing, duplicate, and out-of-range
`input_index`, plus automatic responses:

```json
{"visible_person_count": 2, "keywords": ["people"]}
```

which must raise `InvalidKeywordExtraction`, while count `4` is accepted.

- [ ] **Step 3: Run Claude tests and verify RED**

Run:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m pytest tests/test_claude_vision.py -q
```

Expected: failures show the old global manual schema and missing validation.

- [ ] **Step 4: Add provider-neutral per-target diagnostics types**

In `keyword_extractor.py` add:

```python
@dataclass(frozen=True)
class TargetKeywordExtraction:
    input_index: int
    source_keyword: str
    keyword: str
    occluders: tuple[str, ...] = ()


@dataclass
class KeywordExtractionResult:
    keywords: list[str]
    occluders: list[str] = field(default_factory=list)
    target_results: tuple[TargetKeywordExtraction, ...] = ()
    visible_person_count: int | None = None
```

Export `TargetKeywordExtraction` from `backend.services`.

- [ ] **Step 5: Replace the manual Claude schema and strengthen prompts**

Automatic schema must require `visible_person_count` and `keywords`. Manual
schema must require `visible_person_count` and `target_results`.

The common prompt must explicitly state:

```text
Never exclude an object because it is tiny or incidental when it genuinely
occludes a selected target.
```

Manual prompt must require one indexed result per input target, exhaustive
per-target inspection, strongest-to-weakest occluder ordering, root-parent
person labels, and no arbitrary target generalization.

Both prompts must state the one-to-three versus more-than-three people rule.

- [ ] **Step 6: Parse and validate indexed target results**

In manual mode:

1. Validate `visible_person_count`.
2. Validate that indexes equal `range(len(target_keywords))` exactly once.
3. Select final targets with `validate_refined_target()`.
4. Normalize each per-target occluder array independently.
5. Validate `people` only on occluder arrays, never on manual targets.
6. Union arrays with `union_ranked_occluders()`.
7. Return flat targets/occluders plus per-target diagnostics.

In automatic mode, validate `people` across the returned `keywords`.

- [ ] **Step 7: Run Claude and service tests and verify GREEN**

Run:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m pytest tests/test_prompt_refinement.py tests/test_keyword_extractor.py tests/test_claude_vision.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 2**

```powershell
git add backend/services/keyword_extractor.py backend/services/claude_vision.py backend/services/__init__.py tests/test_claude_vision.py
git commit -m "feat: inspect occluders per target"
```

---

### Task 3: Independent target and occluder quotas at the API boundary

**Files:**
- Modify: `backend/config.yaml`
- Modify: `backend/main.py`
- Modify: `tests/test_keyword_extractor.py`
- Modify: `tests/test_main_claude_keywords.py`

**Interfaces:**
- Consumes: `KeywordExtractionResult.keywords` and `.occluders`
- Adds config: `vlm.max_occluders: 10`
- Preserves: `resolve_keywords(...) -> list[str]`

- [ ] **Step 1: Write failing separate-quota tests**

Add `max_occluders: 10` to the VLM config merge test. Replace the existing
combined-truncation expectation with:

```python
targets = [f"target-{index}" for index in range(10)]
occluders = [f"occluder-{index}" for index in range(10)]
assert await resolve_keywords(...) == [*targets, *occluders]
```

Add a duplicate test where an occluder matches a target case-insensitively and
is removed without consuming an occluder slot.

- [ ] **Step 2: Run API keyword tests and verify RED**

Run:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m pytest tests/test_keyword_extractor.py tests/test_main_claude_keywords.py -q
```

Expected: old logic truncates the combined result to `max_keywords`.

- [ ] **Step 3: Add `max_occluders` and remove combined truncation**

In `backend/config.yaml`:

```yaml
vlm:
  max_keywords: 10
  max_occluders: 10
```

In `resolve_keywords()` normalize targets with `max_keywords`, normalize
occluders with `max_occluders`, remove occluders already present in targets,
and return their concatenation without slicing the combined list.

- [ ] **Step 4: Add concise fallback/truncation logging**

Log target count and occluder count at INFO. Log per-target semantic fallback
and occluder truncation inside the Claude adapter without logging image data.

- [ ] **Step 5: Run Task 3 tests and verify GREEN**

Run:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m pytest tests/test_keyword_extractor.py tests/test_main_claude_keywords.py tests/test_claude_vision.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 3**

```powershell
git add backend/config.yaml backend/main.py tests/test_keyword_extractor.py tests/test_main_claude_keywords.py
git commit -m "fix: preserve the occluder keyword quota"
```

---

### Task 4: Documentation and regression verification

**Files:**
- Modify: `implementation_claude_plan.md`
- Verify: `docs/superpowers/specs/2026-07-30-prompt-refinement-occluder-recall-design.md`

**Interfaces:**
- Documents the final structured response, target validation, quota, and person-cardinality rules
- Makes no runtime changes

- [ ] **Step 1: Update current VLM workflow documentation**

Replace the obsolete global manual output example with the indexed
`target_results` schema. Document:

- deterministic target fallback;
- controlled aliases;
- per-target occluder inspection and ranked union;
- tiny-occluder exception;
- person-parent hierarchy;
- `people` allowed only above three people in auto/occluder output;
- separate `10 + 10` quotas.

- [ ] **Step 2: Run focused regression tests**

Run:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m pytest tests/test_prompt_refinement.py tests/test_keyword_extractor.py tests/test_claude_vision.py tests/test_main_claude_keywords.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run syntax and diff checks**

Run:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m compileall -q backend tests
git diff --check
```

Expected: both commands exit zero.

- [ ] **Step 4: Run the full suite and compare with baseline**

Run:

```powershell
& 'C:\Users\admin\anaconda3\envs\layer\python.exe' -m pytest -q
```

Expected: no failures beyond the 15 documented baseline failures present at
commit `5c0e9b6`. Record exact pass/fail counts and compare failure node IDs.

- [ ] **Step 5: Commit documentation**

```powershell
git add implementation_claude_plan.md
git commit -m "docs: explain safe prompt refinement"
```

- [ ] **Step 6: Request independent code review**

Review the complete diff from `b89d958` through the final implementation HEAD
for semantic-preservation bypasses, missing target coverage, quota regressions,
person-cardinality edge cases, and accidental downstream pipeline changes.
