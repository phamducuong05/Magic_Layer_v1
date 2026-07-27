"""Stable, workflow-ordered filenames for reconstruction debug artifacts."""

from pathlib import Path


_ARTIFACT_ORDER = (
    "initial_modal_mask",
    "completed_amodal_mask",
    "completion_hole",
    "amodal_bbox_mask",
    "target_bbox_mask",
    "accepted_target_bbox",
    "directional_seed",
    "filtered_reconstruction_mask",
    "composition_mask",
    "source",
    "roi_real_pixels",
    "target_modal_protected",
    "occluder_mask",
    "full_occluder_in_roi",
    "foreign_modal_inside_bbox",
    "foreign_modal_outside_bbox",
    "replacement_domain",
    "replaceable_foreign_inside",
    "protected_foreign_inside",
    "foreign_protection",
    "protected_source_pixels",
    "accepted_model_rgb_mask",
    "generation_before_dilation",
    "generation_after_dilation",
    "generation_mask",
    "base_output_512",
    "sr_output",
    "model_output",
    "validated_output",
    "birefnet_masked_input",
    "raw_birefnet_alpha",
    "target_connection_anchor",
    "birefnet_candidate",
    "changed_by_model",
    "generation_evidence",
    "accepted_target_component",
    "reconstruction_extension_alpha",
    "reconstruction_extension_mask",
    "reconstruction_write_alpha",
    "reconstruction_write_mask",
    "composed_reconstruction",
    "final_reconstruction_support",
)

_ARTIFACT_FILENAMES = {
    name: f"{index:02d}_{name}.png"
    for index, name in enumerate(_ARTIFACT_ORDER, start=1)
}


def artifact_filename(name: str) -> str:
    """Return the stable numbered filename for one known artifact."""
    try:
        return _ARTIFACT_FILENAMES[name]
    except KeyError as exc:
        raise ValueError(f"unknown reconstruction artifact {name!r}") from exc


def artifact_path(directory: str | Path, name: str) -> Path:
    """Return a numbered artifact path inside one per-object directory."""
    return Path(directory) / artifact_filename(name)
