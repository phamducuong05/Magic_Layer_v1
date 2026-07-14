"""Static checks for the staged partial-pipeline notebook."""

import json
from pathlib import Path


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "partial_pipeline_v1.ipynb"
)


def _load_cells():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return notebook["cells"]


def test_notebook_code_cells_compile_and_are_explained():
    cells = _load_cells()
    assert cells
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        assert index > 0
        assert cells[index - 1]["cell_type"] == "markdown"
        compile(
            "".join(cell["source"]),
            f"partial_pipeline_v1-cell-{index}",
            "exec",
        )


def test_notebook_matches_current_process_image_stage_order():
    source = "\n".join(
        "".join(cell["source"])
        for cell in _load_cells()
        if cell["cell_type"] == "code"
    )
    calls = [
        "_extract_objects(image, keywords)",
        "_link_overlap_partners(objects)",
        "_complete_overlapping_objects(image, objects)",
        "assign_pair_roles(overlap_pairs, hole_areas)",
        "_apply_pair_decisions(objects, pair_decisions)",
        "_build_reconstruction_masks(objects, kernel_size)",
        "_refine_masks(image_np, raw_masks)",
        "_extract_object_layers(",
        "_generate_final_background(",
    ]
    positions = [source.index(call) for call in calls]
    assert positions == sorted(positions)
    assert "from backend.image_processor import" in source
    assert "tests/test_data/sample.jpg" not in source
    assert "Image.new(" not in source


def test_notebook_documents_current_reconstruction_boundary():
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in _load_cells()
        if cell["cell_type"] == "markdown"
    ).lower()
    assert "hidden rgb reconstruction" in markdown
    assert "not implemented" in markdown
    assert "modal" in markdown
