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
        "extract_objects(image, keywords, segmentation_processor)",
        "link_overlap_partners(objects)",
        "get_completion_candidates(objects)",
        "complete_objects(",
        "assign_pair_roles(overlap_pairs, hole_areas)",
        "apply_pair_decisions(objects, pair_decisions)",
        "build_reconstruction_masks(objects, kernel_size)",
        "refine_objects(image_np, objects, matte)",
        "extract_object_layers(",
        "generate_final_background(",
    ]
    positions = [source.index(call) for call in calls]
    assert positions == sorted(positions)
    assert 'config.get_pipeline_config("completion")' in source
    assert "max_area_growth_ratio=" in source
    assert "max_bbox_growth_ratio=" in source
    assert "from backend.pipeline.segmentation import extract_objects" in source
    assert "from backend.pipeline.completion import" in source
    assert "from backend.pipeline.reconstruction import" in source
    assert "from backend.pipeline.matting import refine_objects" in source
    assert "from backend.pipeline.layers import extract_object_layers" in source
    assert "from backend.pipeline.background import" in source
    assert "from backend.image_processor import" not in source
    assert "_extract_objects" not in source
    assert "_refine_masks" not in source
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
