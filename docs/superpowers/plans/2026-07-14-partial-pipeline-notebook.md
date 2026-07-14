# Partial Pipeline V1 Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the outdated notebook with a top-to-bottom, visually debuggable execution of the workflow currently implemented by `backend.image_processor.process_image()`.

**Architecture:** Call the production orchestration helpers in their existing order so intermediate state is visible without monkey-patching or running inference twice. Keep plotting and base64 decoding local to the notebook. Hidden RGB reconstruction stays explicitly outside the executable path.

**Tech Stack:** Jupyter Notebook JSON, Python, PIL, NumPy, OpenCV, Matplotlib, PyTorch, SAM3, SDAmodal/DIFT, BiRefNet, configured inpainting adapter, pytest.

## Global Constraints

- Modify `examples/partial_pipeline_v1.ipynb`; do not change production behavior.
- Preserve lazy model loading and no-overlap SDAmodal bypass.
- Use original RGB plus modal masks for the current BiRefNet path.
- Visualize reconstruction masks without calling them reconstructed RGB.
- Put explanatory Markdown immediately before every code cell.
- Do not load heavyweight models during local structural verification.
- Preserve unrelated user changes in DIFT and `tests/test_run_mask_completion.py`.

---

### Task 1: Add a notebook regression test

**Files:**
- Create: `tests/test_partial_pipeline_notebook.py`
- Inspect: `examples/partial_pipeline_v1.ipynb`

**Interfaces:**
- Consumes: standard notebook JSON.
- Produces: static checks for syntax, explanations, imports, and helper order.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
C:\Users\admin\anaconda3\envs\layer\python.exe -m pytest tests/test_partial_pipeline_notebook.py -q
```

Expected: the workflow-order test fails because the current notebook stops
before BiRefNet, layer extraction, and final-background generation.

- [ ] **Step 3: Commit only the test if authorized**

```powershell
git add -- tests/test_partial_pipeline_notebook.py
git commit -m "test: define partial pipeline notebook workflow"
```

If authorization is declined, continue without staging unrelated files.

---

### Task 2: Build the staged notebook

**Files:**
- Modify: `examples/partial_pipeline_v1.ipynb`
- Test: `tests/test_partial_pipeline_notebook.py`

**Interfaces:**
- Consumes: `image_path: Path`, `keywords: list[str]`, backend configuration.
- Produces: `objects`, `overlap_pairs`, `pair_decisions`, `soft_alphas`, `layers`, `background`, and `result` from one staged run.

- [ ] **Step 1: Add Setup & Imports**

Resolve the root from either the repository root or `examples/`; import
`config`, `_calc_kernel_size`, `_image_to_base64`, `expand_mask`,
`assign_pair_roles`, `ProcessResult`, and these helpers from
`backend.image_processor`: `_extract_objects`, `_link_overlap_partners`,
`_complete_overlapping_objects`, `_apply_pair_decisions`,
`_build_reconstruction_masks`, `_refine_masks`, `_extract_object_layers`, and
`_generate_final_background`.

Define `ALPHA_THRESHOLD = 0.005`, a `show_mask_overlay()` Matplotlib helper,
and this decoder:

```python
def decode_png_base64(value):
    return Image.open(io.BytesIO(base64.b64decode(value))).copy()
```

- [ ] **Step 2: Add Model Configuration**

Print `config.device` and `config.get_model_config(category)["name"]` for
segmentation, completion, matting, and inpainting. Do not call any
`model_manager.get_*_model()` method here; explain lazy first-use loading.

- [ ] **Step 3: Add Data Loading**

```python
image_path = PROJECT_ROOT / "assets" / "images" / "composite.png"
keywords = ["men", "women"]
keywords = [keyword.strip() for keyword in keywords if keyword.strip()]
if not keywords:
    raise ValueError("keywords must contain at least one non-empty prompt")
if not image_path.is_file():
    raise FileNotFoundError(f"Input image not found: {image_path}")
image = Image.open(image_path).convert("RGB")
image_np = np.asarray(image, dtype=np.uint8)
width, height = image.size
```

Display the original image, dimensions, and prompts. Never create a dummy
fallback image.

- [ ] **Step 4: Add SAM3 extraction and visualization**

Call `_extract_objects(image, keywords)`. Print each object ID, class, label,
modal area, and grouped bounding box. If empty, raise a clear runtime message
after explaining that production would return the original image. Show every
modal mask and a colored source overlay with `(x, y, width, height)` box.

- [ ] **Step 5: Add cross-class overlap visualization**

Call `_link_overlap_partners(objects)`. Print pairs and partner IDs. Draw each
grouped box and positive-area intersection rectangle; derive intersections
only for plotting and keep production `overlap_pairs` for decisions.

- [ ] **Step 6: Add conditional completion visualization**

Call `_complete_overlapping_objects(image, objects)`. For completed objects,
show modal, amodal, completion hole, and source overlay in one row, including
hole area. If none completed, print that SDAmodal was skipped and continue.

- [ ] **Step 7: Add pair decisions**

```python
hole_areas = {
    detected.object_id: detected.completion_hole_area
    for detected in objects
    if detected.completion_hole_area is not None
}
pair_decisions = assign_pair_roles(overlap_pairs, hole_areas)
_apply_pair_decisions(objects, pair_decisions)
```

Print both areas and the chosen roles, or `ambiguous`, for every pair.

- [ ] **Step 8: Add reconstruction-mask visualization**

```python
kernel_size = _calc_kernel_size(image_np)
_build_reconstruction_masks(objects, kernel_size)
reconstruction_objects = [
    obj for obj in objects if obj.reconstruction_mask is not None
]
```

Show each mask and source overlay. The Markdown must state that hidden RGB
reconstruction is not implemented and the later stages still use modal input.

- [ ] **Step 9: Add the current BiRefNet path**

```python
raw_masks = [detected.modal_mask for detected in objects]
labels = [detected.display_label for detected in objects]
soft_alphas = _refine_masks(image_np, raw_masks)
```

Show every grayscale alpha and original-image alpha overlay with min/max values.

- [ ] **Step 10: Add RGBA layer extraction**

```python
layers = _extract_object_layers(
    image, image_np, soft_alphas, labels, kernel_size
)
decoded_layers = [decode_png_base64(layer.png_base64) for layer in layers]
```

Display every RGBA crop and print keyword, offset, width, and height. Explain
that the helper performs component-background inpainting and refinement.

- [ ] **Step 11: Add final-background debugging**

```python
visible_union = np.logical_or.reduce([mask > 0 for mask in raw_masks])
for alpha in soft_alphas:
    visible_union |= alpha > ALPHA_THRESHOLD
expanded_union = expand_mask(visible_union, kernel_size).astype(bool)
background = _generate_final_background(
    image, raw_masks, soft_alphas, kernel_size
)
```

Show the visible union, expanded mask, source overlay, and final background.
Label the union calculation as a diagnostic reproduction of production logic.

- [ ] **Step 12: Add summary and guarded export**

```python
result = ProcessResult(
    background_base64=_image_to_base64(background),
    original_width=width,
    original_height=height,
    layers=layers,
)
```

Print dimensions and layer metadata. Add `EXPORT_OUTPUTS = False`; when enabled,
save `background.png` and decoded RGBA layers under
`examples/out/partial_pipeline_v1` using sanitized labels.

- [ ] **Step 13: Verify GREEN**

Run the focused pytest command from Task 1. Expected: `3 passed`, without model
initialization.

- [ ] **Step 14: Commit only notebook/test if authorized**

```powershell
git add -- examples/partial_pipeline_v1.ipynb tests/test_partial_pipeline_notebook.py
git commit -m "docs: expand partial pipeline notebook"
```

---

### Task 3: Verify the completed change

**Files:**
- Verify: `examples/partial_pipeline_v1.ipynb`
- Verify: `tests/test_partial_pipeline_notebook.py`

**Interfaces:**
- Consumes: completed notebook and test.
- Produces: evidence of valid JSON, compilable cells, correct ordering, and no deterministic regressions.

- [ ] **Step 1: Validate notebook JSON**

```powershell
C:\Users\admin\anaconda3\envs\layer\python.exe -m json.tool examples/partial_pipeline_v1.ipynb > $null
```

Expected: exit code `0`.

- [ ] **Step 2: Run all deterministic tests**

```powershell
C:\Users\admin\anaconda3\envs\layer\python.exe -m pytest -q
```

Expected: all tests pass without heavyweight model loading.

- [ ] **Step 3: Inspect only the scoped diff**

```powershell
git diff --check -- examples/partial_pipeline_v1.ipynb tests/test_partial_pipeline_notebook.py
git status --short
```

Expected: no whitespace errors. Existing DIFT and run-mask-completion test
changes remain untouched.

- [ ] **Step 4: Perform server/GPU acceptance**

Open the notebook with the same environment as the backend, restart the kernel,
and run all cells. Confirm visible source image, grouped masks and boxes,
overlaps, completion outputs, pair decisions, reconstruction masks when
applicable, modal alpha mattes, RGBA layers, final removal mask, and final
background.
