# BiRefNet Alpha Coverage Fallback Design

## Goal

Prevent a SAM3-detected object from disappearing when final group matting with
BiRefNet returns an empty or materially incomplete alpha.

The change is limited to final object matting and layer extraction. Existing
SAM3 segmentation, same-class and cross-class grouping, amodal completion,
object reconstruction, and background inpainting behavior remain unchanged.

## Current Failure

`refine_objects()` clips the BiRefNet alpha to a dilated support region but does
not require the alpha to cover each member of the group. A zero or incomplete
alpha can therefore reach `extract_object_layers()`, where the group can be
skipped as `empty_refined_alpha`.

A group-level coverage ratio is insufficient: a large member can hide the fact
that BiRefNet completely omitted a smaller member.

## Required Fallback Domain

Build a fallback domain independently for every member.

For a member without accepted reconstruction:

```text
fallback_domain = modal_mask
```

For a member with accepted reconstruction:

```text
fallback_domain =
    modal_mask
    OR (
        reconstruction_replacement_domain_mask
        AND reconstruction_accepted_rgb_mask
    )
```

The modal mask guarantees preservation of the visible SAM3 object. The
replacement domain includes the completed hole, while the accepted RGB mask
prevents protected foreign pixels from entering the fallback alpha.

If either reconstruction mask is unavailable or has an incompatible shape,
the member falls back to its modal mask only.

## Validation and Recovery

After BiRefNet alpha normalization and support clipping, validate coverage for
each member against its fallback domain:

```text
coverage_ratio =
    pixels(alpha > alpha_presence_threshold AND fallback_domain)
    / pixels(fallback_domain)
```

Fallback is required when:

- the alpha contains no finite usable values;
- no alpha pixel above the presence threshold intersects the member domain; or
- member coverage is below the configured minimum.

Only failed members are recovered:

```text
validated_alpha = maximum(
    birefnet_alpha,
    union(fallback_domain for each failed member)
)
```

Successful members retain their BiRefNet soft alpha. Fallback pixels are hard
alpha (`1.0`) so a confirmed object cannot disappear.

## Final Invariant

Color-based alpha refinement and cleanup can modify the validated alpha.
Before creating the RGBA layer, run the same per-member coverage check again.
If a member is missing, reapply only that member's fallback domain.

This final guard must run before bounding-box extraction so
`empty_refined_alpha` cannot discard a group that still has a valid fallback
domain.

## Configuration

Add narrowly scoped matting settings:

```yaml
matting:
  alpha_presence_threshold: 0.05
  min_member_alpha_coverage_ratio: 0.95
```

Both values must be finite and within `[0, 1]`. Existing matting defaults and
thresholds remain unchanged.

## Diagnostics

Log one decision per failed member with:

- group ID and member ID;
- measured coverage ratio;
- fallback-domain pixel count;
- stage (`birefnet_output` or `final_alpha`);
- reason for fallback.

The normal accepted path should log no warning and preserve the model alpha.

## Tests

Focused tests cover:

1. A standalone modal object survives an empty BiRefNet alpha.
2. In a multi-member group, only the omitted member receives hard fallback.
3. A reconstructed member uses modal pixels plus the safe accepted part of its
   replacement domain.
4. Protected foreign pixels outside the accepted RGB mask remain transparent.
5. A sufficiently covered BiRefNet result is not changed.
6. Final layer refinement cannot remove a member restored by the fallback.

Only the focused matting and layer tests are required for this change.
