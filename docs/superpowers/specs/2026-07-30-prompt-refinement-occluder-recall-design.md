# Prompt Refinement and Occluder Recall Design

## 1. Goal

Improve the VLM keyword-refinement stage without changing the existing image
pipeline after keyword resolution.

The revised stage must:

- Preserve the semantic identity and cardinality of user-supplied targets.
- Simplify long target descriptions only through controlled vocabulary rules.
- Inspect every manual target independently for occluders.
- Include a real occluder even when it is tiny or otherwise incidental.
- Keep body parts, clothing, wearables, and accessories attached to their
  root-level parent object.
- Avoid generalizing a small number of visible people into `people`.
- Preserve the existing segmentation, grouping, completion, reconstruction,
  matting, layer extraction, and background-inpainting workflows.

## 2. Current Problems

The current manual-target prompt explicitly authorizes semantic replacement,
including examples such as `human individual` becoming `person`. The response
contains one global target array and one global occluder array, so there is no
way to verify that every target was inspected.

The common prompt rules also exclude tiny objects, body parts, clothing,
wearables, accessories, and container contents without making the tiny-object
exception sufficiently authoritative for confirmed occluders.

Finally, `resolve_keywords()` truncates the combined target and occluder list
to one ten-keyword budget. Targets are placed first, so valid occluders can be
silently removed.

## 3. Semantic Preservation Contract

User input is the source of truth.

### 3.1 Invariants

- A simple, single-token target passes through unchanged apart from trimming.
- The system must preserve object type, specificity, and grammatical number.
- `man` must not become `person` or `people`.
- `boy` must not become `child`.
- `cars` must not become `car`.
- A user-supplied `people` remains `people`, regardless of the number of
  people visible in the image.
- Case-insensitive duplicate removal remains permitted and preserves the first
  spelling and order supplied by the user.

### 3.2 Controlled simplification

Long descriptions may be shortened by removing determiners, colors,
materials, styles, states, and descriptive clauses. For example:

- `tall man in red` becomes `man`.
- `red four-door passenger automobile` becomes `car`.

Semantic replacement is accepted only when it is supported by a code-owned
simple-object vocabulary or alias map. Example aliases include
`automobile -> car`. The SAM3 BPE file
`bpe_simple_vocab_16e6.txt.gz` is a tokenizer vocabulary and must not be used
as an object-label allow-list.

The VLM may propose a refinement, but a deterministic validator owns the final
decision:

1. Single-token inputs always use the original token.
2. A multi-token candidate may use a preserved head noun found in the source.
3. A canonical replacement must match an explicit alias for a word or phrase
   present in the source.
4. Singular/plural cardinality must remain unchanged.
5. Any candidate that fails validation falls back to the normalized original
   target without failing the whole request.

The vocabulary and alias map must be small, auditable, and covered by tests.
Unknown phrases fall back to the original phrase rather than being guessed.

## 4. Per-Target Structured Output

Manual mode uses one VLM call with output organized by input target:

```json
{
  "visible_person_count": 2,
  "target_results": [
    {
      "input_index": 0,
      "refined_keyword": "car",
      "occluders": ["man", "tree"]
    },
    {
      "input_index": 1,
      "refined_keyword": "table",
      "occluders": ["man", "chair"]
    }
  ]
}
```

`input_index` is the only identity accepted from the model. The backend must
receive exactly one result for every supplied target. Missing, duplicate, or
out-of-range indexes make the structured response invalid rather than
silently skipping a target.

The final target keyword is selected by the semantic validator in Section 3,
not trusted directly from `refined_keyword`.

Automatic mode may keep a global keyword result, but it also reports
`visible_person_count` so the people-cardinality contract can be audited and
validated.

## 5. Occluder Recall Contract

The model must exhaustively inspect each target and return every independent
foreground object that visibly covers, overlaps, lies on top of, or blocks
any part of that target.

### 5.1 Tiny-object exception

The normal exclusion for tiny or incidental objects does not apply once an
object is identified as a real occluder. This exception is authoritative in
both automatic and manual modes.

An object that is merely nearby and does not overlap a target is not an
occluder.

### 5.2 Root-parent hierarchy

Body parts, clothing, footwear, wearables, and accessories do not become
independent keywords:

- A hand, glasses, hat, or shirt associated with a person resolves to the
  person parent.
- When visual evidence supports a specific person class, use `man`, `woman`,
  `boy`, or `girl`.
- When age or gender cannot be determined confidently, use `person`.
- Do not infer demographic attributes from weak or ambiguous evidence.

Existing exclusions for background scenery, architecture, reflections,
printed images, and uncertain non-occluding objects remain in force.

## 6. People Cardinality

This rule applies only to automatic extraction and detected occluders. It does
not rewrite user-supplied targets.

- With one to three visible people, the model must not return `people`.
- For one to three people, use `man`, `woman`, `boy`, or `girl` only when the
  image provides sufficient evidence; otherwise use `person`.
- With more than three visible people, `people` is allowed as a grouped
  segmentation keyword.
- One semantic keyword may still yield multiple raw SAM3 masks. For example,
  `man` may segment two visible men.

If a structured response reports three or fewer visible people but returns
`people` as an automatic or occluder keyword, the response violates the
contract. It must not silently enter the segmentation pipeline.

## 7. Occluder Union

Per-target occluder lists are retained for diagnostics, then combined into the
flat keyword list needed by SAM3.

For example:

```text
car   <- man, tree
table <- man, chair
```

produces:

```text
targets:   car, table
occluders: man, tree, chair
```

Union behavior:

1. Normalize parent-object labels before union.
2. Deduplicate case-insensitively while preserving first occurrence.
3. Remove occluder keywords already present in the final target list.
4. Preserve target order first.
5. Require each per-target occluder array to be ordered from strongest to
   weakest visible coverage.
6. Flatten those arrays by rank and then input target index: first take the
   strongest occluder for each target in target order, then the second
   strongest for each target, and so on. This prevents one target with many
   occluders from consuming the complete quota before another target is
   represented.

This is a keyword-list union only. It does not merge masks or objects and does
not determine geometric occlusion. Existing downstream overlap, containment,
completion, and reconstruction logic remains authoritative after SAM3
produces raw masks.

## 8. Independent Quotas

Use separate safety limits:

- At most 10 user targets.
- At most 10 unique occluder keywords.
- The final SAM3 input may therefore contain up to 20 keywords.

Targets never consume the occluder quota. If the VLM returns more than ten
unique occluders, keep the ten strongest occluders according to the stable
ordering in Section 7 and record truncation in diagnostics.

## 9. Error Handling and Diagnostics

- Malformed JSON or a structurally invalid response continues to raise
  `InvalidKeywordExtraction`.
- Missing, duplicate, or out-of-range manual `input_index` values are
  structural failures.
- A semantically invalid target refinement falls back only that target to the
  normalized original input.
- Invalid `people` cardinality in automatic or occluder output is a response
  failure because the backend cannot safely invent specific replacement
  labels.
- Logs record target count, unique occluder count, truncation, and target
  refinement fallbacks.
- Logs must not contain image bytes or base64 image data.

Existing HTTP mappings remain unchanged:

- Invalid supplied input returns HTTP 400.
- Unavailable VLM service returns HTTP 503.
- Unusable VLM structured output returns HTTP 502.

## 10. Integration Boundaries

Changes are limited to:

- The Claude Vision prompt and structured-output schemas.
- Provider-neutral extraction result types when per-target diagnostics require
  them.
- Controlled target-refinement validation.
- Keyword resolution and separate quota handling.
- Focused tests and documentation.

The implementation must not alter:

- SAM3 segmentation internals.
- Same-class or cross-class grouping.
- Bounding-box containment planning.
- Amodal completion.
- Occlusion direction decisions.
- Object reconstruction.
- Matting, layer extraction, or background inpainting.

## 11. Test Strategy

Tests are written first and run with the Conda `layer` environment.

Required regression coverage:

- Manual `man -> people` is rejected and the final target remains `man`.
- Manual `people` remains `people` even when the response reports two people.
- `tall man in red -> man` is accepted.
- Explicit alias `automobile -> car` is accepted.
- Singular/plural changes are rejected.
- Tiny-occluder exception appears in both automatic and manual prompt
  contracts.
- Manual output contains exactly one result per input index.
- Missing, duplicate, and out-of-range indexes are rejected.
- Occluders from multiple targets are unioned and deduplicated.
- Ten targets plus ten occluders are not truncated to ten total.
- Body parts and wearables resolve to a person parent in the prompt contract.
- One to three auto-detected people cannot produce `people`.
- More than three people may produce `people`.
- Existing API error mapping and downstream pipeline entry contracts remain
  unchanged.

## 12. Success Criteria

The change is successful when:

- Manual targets cannot be arbitrarily generalized.
- Every manual target is explicitly represented in the VLM response.
- Tiny confirmed occluders are not excluded because of size.
- Valid occluders are no longer silently displaced by target quota.
- Small groups of people are not generalized to `people`.
- Existing downstream computer-vision workflows receive only a validated flat
  keyword list and otherwise behave exactly as before.
