"""Deterministic policy for VLM-proposed SAM3 keyword refinements."""

from __future__ import annotations

from collections.abc import Sequence

from .keyword_extractor import InvalidKeywordExtraction

SIMPLE_OBJECT_VOCAB = frozenset(
    {
        "boy",
        "boys",
        "car",
        "cars",
        "chair",
        "chairs",
        "girl",
        "girls",
        "man",
        "men",
        "person",
        "persons",
        "table",
        "tables",
        "woman",
        "women",
    }
)

SIMPLE_VOCAB_ALIASES = {
    "automobile": "car",
    "automobiles": "cars",
}

PERSON_CHILD_LABELS = frozenset(
    {
        "clothes",
        "clothing",
        "coat",
        "coats",
        "dress",
        "dresses",
        "glasses",
        "hand",
        "hands",
        "hat",
        "hats",
        "jacket",
        "jackets",
        "pants",
        "shirt",
        "shirts",
        "shoe",
        "shoes",
    }
)


def _normalized_words(value: str) -> tuple[str, tuple[str, ...]]:
    normalized = " ".join(value.split())
    return normalized, tuple(word.casefold() for word in normalized.split())


def validate_refined_target(source: str, candidate: str) -> str:
    """Accept only source-preserving heads or explicitly controlled aliases."""
    normalized_source, source_words = _normalized_words(source)
    normalized_candidate, candidate_words = _normalized_words(candidate)
    if len(source_words) <= 1:
        return normalized_source
    if normalized_candidate.casefold() == normalized_source.casefold():
        return normalized_source

    candidate_key = normalized_candidate.casefold()
    source_concepts = {
        word for word in source_words if word in SIMPLE_OBJECT_VOCAB
    }
    source_text = " ".join(source_words)
    for alias_source, canonical in SIMPLE_VOCAB_ALIASES.items():
        alias_words = tuple(alias_source.casefold().split())
        if len(alias_words) == 1:
            alias_present = alias_words[0] in source_words
        else:
            alias_present = (
                f" {' '.join(alias_words)} " in f" {source_text} "
            )
        if alias_present:
            source_concepts.add(canonical.casefold())

    if (
        len(candidate_words) == 1
        and len(source_concepts) == 1
        and candidate_key in source_concepts
    ):
        return normalized_candidate

    return normalized_source


def validate_root_person_labels(keywords: Sequence[str]) -> None:
    """Reject person parts and wearables that should use a parent label."""
    invalid = next(
        (
            keyword
            for keyword in keywords
            if keyword.strip().casefold() in PERSON_CHILD_LABELS
        ),
        None,
    )
    if invalid is not None:
        raise InvalidKeywordExtraction(
            f"Claude returned {invalid!r} instead of a root person label."
        )


def validate_people_keyword_usage(
    keywords: Sequence[str],
    visible_person_count: int,
) -> None:
    """Reject invalid counts and small-group generalization to ``people``."""
    if (
        isinstance(visible_person_count, bool)
        or not isinstance(visible_person_count, int)
        or visible_person_count < 0
    ):
        raise InvalidKeywordExtraction(
            "Claude returned an invalid visible person count."
        )
    if visible_person_count <= 3 and any(
        keyword.strip().casefold() == "people" for keyword in keywords
    ):
        raise InvalidKeywordExtraction(
            "Claude generalized three or fewer visible people to 'people'."
        )


def union_ranked_occluders(
    per_target: Sequence[Sequence[str]],
    *,
    target_keywords: Sequence[str],
    max_occluders: int,
) -> list[str]:
    """Union ranked lists fairly so every target can contribute an occluder."""
    if max_occluders < 0:
        raise ValueError("max_occluders must be non-negative")

    excluded = {
        target.strip().casefold()
        for target in target_keywords
        if target.strip()
    }
    seen = set(excluded)
    cursors = [0] * len(per_target)
    result: list[str] = []

    while len(result) < max_occluders:
        made_progress = False
        for target_index, occluders in enumerate(per_target):
            while cursors[target_index] < len(occluders):
                value = occluders[cursors[target_index]]
                cursors[target_index] += 1
                normalized = " ".join(value.split())
                key = normalized.casefold()
                if not normalized or key in seen:
                    continue
                seen.add(key)
                result.append(normalized)
                made_progress = True
                break
            if len(result) == max_occluders:
                break
        if not made_progress:
            break

    return result
