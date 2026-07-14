"""Load the configured SDAmodal model for production inference."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def _completion_model_classes() -> Mapping[str, type]:
    from . import models

    return vars(models)


def load_sdamodal_model(
    config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    device: str,
    model_classes: Mapping[str, type] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Construct, restore, and activate the configured SDAmodal model."""
    with Path(config_path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    model_config = config["model"]
    algorithm = model_config["algo"]
    available_models = (
        model_classes
        if model_classes is not None
        else _completion_model_classes()
    )
    model_class = available_models.get(algorithm)
    if model_class is None:
        raise ValueError(
            f"SDAmodal algorithm '{algorithm}' is not available"
        )

    model = model_class(
        model_config,
        dist_model=False,
        device=device,
    )
    model.load_state(checkpoint_path)
    model.switch_to("eval")
    return model, config
