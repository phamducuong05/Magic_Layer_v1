from types import SimpleNamespace


def test_reconstruction_pair_labels_use_human_readable_component_names():
    from backend.pipeline import orchestrator

    objects = [
        SimpleNamespace(object_id="object-0", display_label="dog"),
        SimpleNamespace(object_id="object-1", display_label="wooden chair"),
    ]
    decisions = [
        SimpleNamespace(
            reconstruction_directions=(("object-0", "object-1"),),
            occluded_id="object-0",
            occluder_id="object-1",
        )
    ]

    assert orchestrator._reconstruction_pair_labels(objects, decisions) == [
        ("dog", "wooden chair")
    ]
