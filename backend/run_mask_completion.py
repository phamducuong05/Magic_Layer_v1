"""Run and visualize raw SAM3 masks through validated amodal completion.

This is a diagnostic command. Its same-class grouping output previews the
planned post-reconstruction grouping policy without changing production state.
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw


logger = logging.getLogger(__name__)

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COLORS = [
    (239, 83, 80),
    (66, 165, 245),
    (102, 187, 106),
    (255, 167, 38),
    (171, 71, 188),
    (38, 198, 218),
    (255, 238, 88),
    (141, 110, 99),
]


@dataclass
class DiagnosticGroup:
    """One diagnostic-only same-class bbox group."""

    semantic_class: str
    member_ids: list[str]
    mask: np.ndarray
    bbox: tuple[int, int, int, int]


class _RecordingCompletionModel:
    """Record raw model outputs while preserving the production model API."""

    def __init__(self, model: Any):
        self.model = model
        self.outputs: Any = None

    def complete(self, image, modal_masks, bboxes):
        self.outputs = self.model.complete(image, modal_masks, bboxes)
        return self.outputs


def save_masks(
    masks: Sequence[np.ndarray], output_dir: Path
) -> list[Path]:
    """Save masks as binary PNGs and return their paths in input order."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for index, mask in enumerate(masks):
        path = output_dir / f"mask_{index}.png"
        _mask_image(mask).save(path)
        paths.append(path)

    return paths


def _mask_array(mask: Any) -> np.ndarray | None:
    """Return a two-dimensional boolean visualization mask when possible."""
    if not isinstance(mask, np.ndarray):
        return None
    squeezed = np.asarray(mask).squeeze()
    if squeezed.ndim != 2:
        return None
    try:
        return squeezed > 0
    except (TypeError, ValueError):
        return None


def _aligned_mask(mask: Any, size: tuple[int, int]) -> np.ndarray | None:
    """Convert a visualizable mask to the requested image size."""
    binary = _mask_array(mask)
    if binary is None:
        return None
    width, height = size
    if binary.shape != (height, width):
        binary = np.asarray(
            Image.fromarray(binary.astype(np.uint8) * 255, mode="L").resize(
                size, Image.Resampling.NEAREST
            )
        ) > 0
    return binary


def _mask_image(mask: Any) -> Image.Image:
    binary = _mask_array(np.asarray(mask))
    if binary is None:
        raise ValueError("mask must be a two-dimensional numeric array")
    return Image.fromarray(binary.astype(np.uint8) * 255, mode="L")


def _bbox_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    return (
        max(first_x, second_x)
        < min(first_x + first_width, second_x + second_width)
        and max(first_y, second_y)
        < min(first_y + first_height, second_y + second_height)
    )


def _union_bbox(
    bboxes: Sequence[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    x0 = min(bbox[0] for bbox in bboxes)
    y0 = min(bbox[1] for bbox in bboxes)
    x1 = max(bbox[0] + bbox[2] for bbox in bboxes)
    y1 = max(bbox[1] + bbox[3] for bbox in bboxes)
    return x0, y0, x1 - x0, y1 - y0


def build_diagnostic_groups(objects: Sequence[Any]) -> list[DiagnosticGroup]:
    """Preview stable same-class groups using original bbox overlap only."""
    parents = list(range(len(objects)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, first in enumerate(objects):
        for second_index in range(first_index + 1, len(objects)):
            second = objects[second_index]
            if (
                first.semantic_class == second.semantic_class
                and _bbox_overlap(
                    first.original_modal_bbox,
                    second.original_modal_bbox,
                )
            ):
                union(first_index, second_index)

    grouped_indices: dict[int, list[int]] = {}
    for index in range(len(objects)):
        grouped_indices.setdefault(find(index), []).append(index)

    groups: list[DiagnosticGroup] = []
    for indices in grouped_indices.values():
        members = [objects[index] for index in indices]
        merged = np.zeros_like(members[0].modal_mask, dtype=bool)
        for member in members:
            source_mask = (
                member.amodal_mask
                if member.amodal_mask is not None
                else member.modal_mask
            )
            merged |= np.asarray(source_mask) > 0
        groups.append(
            DiagnosticGroup(
                semantic_class=members[0].semantic_class,
                member_ids=[member.object_id for member in members],
                mask=merged,
                bbox=_union_bbox(
                    [member.original_modal_bbox for member in members]
                ),
            )
        )
    return groups


def _overlay(
    image: Image.Image,
    masks: Sequence[Any],
    labels: Sequence[str],
    bboxes: Sequence[tuple[int, int, int, int]] | None = None,
) -> Image.Image:
    """Overlay colored masks, labels, and optional boxes on the source image."""
    canvas = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    aligned_masks = [_aligned_mask(mask, image.size) for mask in masks]
    for index, binary in enumerate(aligned_masks):
        if binary is None:
            continue
        color = np.asarray(COLORS[index % len(COLORS)], dtype=np.float32)
        canvas[binary] = canvas[binary] * 0.55 + color * 0.45

    result = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(result)
    for index, label in enumerate(labels):
        color = COLORS[index % len(COLORS)]
        if bboxes is not None:
            x, y, width, height = bboxes[index]
            draw.rectangle((x, y, x + width - 1, y + height - 1), outline=color, width=2)
            text_position = (x + 2, max(0, y - 12))
        else:
            binary = aligned_masks[index]
            if binary is None or not np.any(binary):
                text_position = (3, 3 + index * 12)
            else:
                ys, xs = np.nonzero(binary)
                text_position = (int(xs.min()), int(ys.min()))
        draw.text(text_position, label, fill=color, stroke_width=2, stroke_fill="black")
    return result


def _pair_overview(
    image: Image.Image,
    objects: Sequence[Any],
    pairs: Sequence[tuple[str, str]],
    title: str,
) -> Image.Image:
    objects_by_id = {detected.object_id: detected for detected in objects}
    masks = [
        detected.amodal_mask
        if detected.amodal_mask is not None
        else detected.modal_mask
        for detected in objects
    ]
    result = _overlay(
        image,
        masks,
        [detected.display_label for detected in objects],
        [detected.original_modal_bbox for detected in objects],
    )
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, max(1, len(title) * 7), 14), fill="black")
    draw.text((2, 2), title, fill="white")
    for index, (first_id, second_id) in enumerate(pairs):
        first = objects_by_id[first_id].original_modal_bbox
        second = objects_by_id[second_id].original_modal_bbox
        first_center = (first[0] + first[2] // 2, first[1] + first[3] // 2)
        second_center = (
            second[0] + second[2] // 2,
            second[1] + second[3] // 2,
        )
        draw.line(
            (first_center, second_center),
            fill=COLORS[index % len(COLORS)],
            width=3,
        )
    return result


def _comparison_image(
    image: Image.Image,
    detected: Any,
    raw_prediction: Any,
) -> Image.Image:
    panels: list[tuple[str, Image.Image]] = []
    masks = [
        ("modal", detected.modal_mask),
        ("raw prediction", raw_prediction),
        (
            "validated/fallback",
            detected.amodal_mask
            if detected.amodal_mask is not None
            else detected.modal_mask,
        ),
        (
            "completion hole",
            detected.completion_hole_mask
            if detected.completion_hole_mask is not None
            else np.zeros_like(detected.modal_mask),
        ),
    ]
    for title, mask in masks:
        binary = _aligned_mask(mask, image.size)
        panel = (
            _overlay(image, [binary], [title])
            if binary is not None
            else image.copy()
        )
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, max(1, len(title) * 7), 14), fill="black")
        draw.text((2, 2), title, fill="white")
        panels.append((title, panel))

    comparison = Image.new("RGB", (image.width * 2, image.height * 2))
    for index, (_, panel) in enumerate(panels):
        comparison.paste(
            panel,
            ((index % 2) * image.width, (index // 2) * image.height),
        )
    return comparison


def save_workflow_visualizations(
    image: Image.Image,
    objects: Sequence[Any],
    potential_pairs: Sequence[tuple[str, str]],
    retained_pairs: Sequence[tuple[str, str]],
    raw_predictions: dict[str, Any],
    groups: Sequence[DiagnosticGroup],
    output_dir: Path,
) -> Path:
    """Save every diagnostic stage and return the JSON summary path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB")
    image.save(output_dir / "00_original.png")

    raw_dir = output_dir / "01_raw_masks"
    raw_dir.mkdir(exist_ok=True)
    for detected in objects:
        _mask_image(detected.modal_mask).save(
            raw_dir / f"{detected.object_id}_mask.png"
        )
        _overlay(
            image,
            [detected.modal_mask],
            [detected.display_label],
            [detected.original_modal_bbox],
        ).save(raw_dir / f"{detected.object_id}_overlay.png")
    _overlay(
        image,
        [detected.modal_mask for detected in objects],
        [detected.display_label for detected in objects],
        [detected.original_modal_bbox for detected in objects],
    ).save(raw_dir / "overview.png")

    bbox_dir = output_dir / "02_cross_class_bbox"
    bbox_dir.mkdir(exist_ok=True)
    _pair_overview(image, objects, potential_pairs, "candidate bbox pairs").save(
        bbox_dir / "candidate_pairs.png"
    )
    objects_by_id = {detected.object_id: detected for detected in objects}
    for first_id, second_id in potential_pairs:
        members = [objects_by_id[first_id], objects_by_id[second_id]]
        _overlay(
            image,
            [member.modal_mask for member in members],
            [member.display_label for member in members],
            [member.original_modal_bbox for member in members],
        ).save(bbox_dir / f"pair_{first_id}_{second_id}.png")

    prediction_dir = output_dir / "03_raw_completion"
    prediction_dir.mkdir(exist_ok=True)
    for detected in objects:
        if detected.object_id not in raw_predictions:
            continue
        prediction = raw_predictions[detected.object_id]
        binary = _mask_array(prediction)
        if binary is not None:
            _mask_image(binary).save(
                prediction_dir / f"{detected.object_id}_predicted.png"
            )
        _comparison_image(image, detected, prediction).save(
            prediction_dir / f"{detected.object_id}_comparison.png"
        )

    validated_dir = output_dir / "04_validated_completion"
    validated_dir.mkdir(exist_ok=True)
    validated_masks: list[np.ndarray] = []
    for detected in objects:
        validated = (
            detected.amodal_mask
            if detected.amodal_mask is not None
            else detected.modal_mask
        )
        hole = (
            detected.completion_hole_mask
            if detected.completion_hole_mask is not None
            else np.zeros_like(detected.modal_mask, dtype=bool)
        )
        validated_masks.append(np.asarray(validated) > 0)
        _mask_image(validated).save(
            validated_dir / f"{detected.object_id}_validated.png"
        )
        _mask_image(hole).save(
            validated_dir / f"{detected.object_id}_completion_hole.png"
        )
    _overlay(
        image,
        validated_masks,
        [detected.display_label for detected in objects],
    ).save(validated_dir / "overview.png")

    filter_dir = output_dir / "05_amodal_pair_filter"
    filter_dir.mkdir(exist_ok=True)
    retained_set = set(retained_pairs)
    rejected_pairs = [pair for pair in potential_pairs if pair not in retained_set]
    _pair_overview(image, objects, retained_pairs, "retained amodal pairs").save(
        filter_dir / "retained_pairs.png"
    )
    _pair_overview(image, objects, rejected_pairs, "rejected amodal pairs").save(
        filter_dir / "rejected_pairs.png"
    )

    group_dir = output_dir / "06_diagnostic_groups"
    group_dir.mkdir(exist_ok=True)
    for index, group in enumerate(groups):
        _mask_image(group.mask).save(group_dir / f"group-{index}_mask.png")
        _overlay(
            image,
            [group.mask],
            [f"{group.semantic_class}: {', '.join(group.member_ids)}"],
            [group.bbox],
        ).save(group_dir / f"group-{index}_overlay.png")
    final_overview = _overlay(
        image,
        [group.mask for group in groups],
        [
            f"group-{index} {group.semantic_class}"
            for index, group in enumerate(groups)
        ],
        [group.bbox for group in groups],
    )
    final_overview.save(group_dir / "overview.png")
    final_overview.save(output_dir / "07_final_overview.png")

    summary = {
        "note": (
            "06_diagnostic_groups previews Step 29 bbox-only grouping; it does "
            "not mutate or replace production pipeline grouping."
        ),
        "objects": [
            {
                "object_id": detected.object_id,
                "semantic_class": detected.semantic_class,
                "display_label": detected.display_label,
                "original_modal_bbox": list(detected.original_modal_bbox),
                "potential_partner_ids": sorted(detected.overlap_partner_ids),
                "raw_prediction_available": detected.object_id in raw_predictions,
                "completion_hole_area": detected.completion_hole_area,
                "validated_equals_modal": bool(
                    np.array_equal(
                        np.asarray(
                            detected.amodal_mask
                            if detected.amodal_mask is not None
                            else detected.modal_mask
                        )
                        > 0,
                        np.asarray(detected.modal_mask) > 0,
                    )
                ),
            }
            for detected in objects
        ],
        "potential_bbox_pairs": [list(pair) for pair in potential_pairs],
        "retained_amodal_pairs": [list(pair) for pair in retained_pairs],
        "rejected_amodal_pairs": [list(pair) for pair in rejected_pairs],
        "diagnostic_groups": [
            {
                "group_id": f"group-{index}",
                "semantic_class": group.semantic_class,
                "member_ids": group.member_ids,
                "bbox": list(group.bbox),
            }
            for index, group in enumerate(groups)
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_path


def run_diagnostics(
    image: Image.Image,
    keywords: Sequence[str],
    output_dir: Path,
    *,
    processor: Any,
    get_completion_model: Callable[[], Any],
    completion_config: dict[str, Any],
) -> Path:
    """Execute the real raw-mask/completion stages and save diagnostics."""
    from backend.pipeline.completion import (
        complete_objects,
        filter_pairs_by_amodal_overlap,
        get_completion_candidates,
        link_overlap_partners,
    )
    from backend.pipeline.segmentation import extract_raw_objects

    image = image.convert("RGB")
    objects = extract_raw_objects(image, keywords, processor)
    potential_pairs = link_overlap_partners(objects)
    candidates = get_completion_candidates(objects)

    raw_predictions: dict[str, Any] = {}
    if candidates:
        recorder = _RecordingCompletionModel(get_completion_model())
        complete_objects(
            image,
            candidates,
            recorder,
            max_area_growth_ratio=float(
                completion_config["max_area_growth_ratio"]
            ),
            max_bbox_growth_ratio=float(
                completion_config["max_bbox_growth_ratio"]
            ),
        )
        if isinstance(recorder.outputs, Sequence):
            raw_predictions.update(
                {
                    detected.object_id: output
                    for detected, output in zip(candidates, recorder.outputs)
                }
            )

    retained_pairs = filter_pairs_by_amodal_overlap(objects, potential_pairs)
    groups = build_diagnostic_groups(objects)
    return save_workflow_visualizations(
        image,
        objects,
        potential_pairs,
        retained_pairs,
        raw_predictions,
        groups,
        output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run raw SAM3 extraction, cross-class overlap detection, SDAmodal "
            "completion/validation, and diagnostic grouping visualizations."
        )
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--keywords",
        required=True,
        help="Comma-separated semantic prompts, for example: person,chair",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    keywords = [
        keyword.strip()
        for keyword in args.keywords.split(",")
        if keyword.strip()
    ]
    if not keywords:
        parser.error("--keywords must contain at least one non-empty prompt")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    from backend.config import config
    from backend.models import model_manager

    processor = model_manager.get_segmentation_model().get_processor()
    completion_config = config.get_pipeline_config("completion")
    with Image.open(args.image) as source:
        summary_path = run_diagnostics(
            source.convert("RGB"),
            keywords,
            args.output_dir,
            processor=processor,
            get_completion_model=model_manager.get_completion_model,
            completion_config=completion_config,
        )

    png_paths = sorted(args.output_dir.rglob("*.png"))
    print(
        f"Saved {len(png_paths)} visualization images to "
        f"{args.output_dir.resolve()}"
    )
    print(f"Summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
