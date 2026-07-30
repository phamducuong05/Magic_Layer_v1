"""Behavioral tests for deterministic prompt-refinement policy."""

import pytest

from backend.services.keyword_extractor import InvalidKeywordExtraction
from backend.services.prompt_refinement import (
    union_ranked_occluders,
    validate_people_keyword_usage,
    validate_refined_target,
)


def test_single_token_target_never_uses_model_generalization():
    assert validate_refined_target("man", "people") == "man"


def test_long_description_can_keep_a_known_head_noun():
    assert validate_refined_target("tall man in red", "man") == "man"


def test_controlled_alias_can_simplify_an_object_name():
    assert (
        validate_refined_target(
            "red four-door passenger automobile",
            "car",
        )
        == "car"
    )


def test_number_changing_candidate_falls_back_to_source():
    assert validate_refined_target("two men", "man") == "two men"


def test_unknown_replacement_falls_back_to_source():
    assert validate_refined_target("wooden cabinet", "furniture") == (
        "wooden cabinet"
    )


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("man beside car", "car"),
        ("man beside car", "man"),
        ("man beside automobile", "car"),
        ("wooden cabinet beside car", "car"),
        ("dog beside automobile", "car"),
    ],
)
def test_multi_object_description_cannot_switch_or_guess_the_target(
    source,
    candidate,
):
    assert validate_refined_target(source, candidate) == source


def test_small_visible_group_cannot_be_generalized_to_people():
    with pytest.raises(InvalidKeywordExtraction, match="people"):
        validate_people_keyword_usage(["people"], 3)


def test_people_is_allowed_above_three_visible_people():
    validate_people_keyword_usage(["people"], 4)


@pytest.mark.parametrize("invalid_count", [-1, True, 1.5, "2"])
def test_people_validation_rejects_invalid_visible_count(invalid_count):
    with pytest.raises(InvalidKeywordExtraction, match="person count"):
        validate_people_keyword_usage(["person"], invalid_count)


def test_ranked_union_represents_each_target_before_lower_ranked_occluders():
    assert union_ranked_occluders(
        [["man", "tree"], ["Man", "chair"]],
        target_keywords=["car", "table"],
        max_occluders=10,
    ) == ["man", "chair", "tree"]


def test_ranked_union_removes_target_duplicates_without_spending_quota():
    assert union_ranked_occluders(
        [["car", "person"], ["tree"]],
        target_keywords=["Car"],
        max_occluders=2,
    ) == ["person", "tree"]
