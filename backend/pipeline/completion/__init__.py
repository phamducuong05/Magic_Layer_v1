"""Cross-class overlap linking and explicit-model amodal completion."""

from .completion import (
    complete_objects,
    filter_pairs_by_amodal_overlap,
    get_completion_candidates,
    link_overlap_partners,
)

__all__ = [
    "complete_objects",
    "filter_pairs_by_amodal_overlap",
    "get_completion_candidates",
    "link_overlap_partners",
]
