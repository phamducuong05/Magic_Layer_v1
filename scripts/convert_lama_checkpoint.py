"""Convert a trusted legacy LaMa Lightning checkpoint to generator tensors."""

from __future__ import annotations

import argparse
import pickle
import types
from pathlib import Path

import torch


class _LegacyMetadataStub:
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._legacy_state = state


class _TrustedLegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("pytorch_lightning.") or module.startswith(
            "omegaconf."
        ):
            return type(
                name,
                (_LegacyMetadataStub,),
                {"__module__": module},
            )
        return super().find_class(module, name)


TRUSTED_LEGACY_PICKLE = types.SimpleNamespace(
    Unpickler=_TrustedLegacyUnpickler,
    load=pickle.load,
    loads=pickle.loads,
    dump=pickle.dump,
    dumps=pickle.dumps,
    __name__="trusted_lama_pickle",
)


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            pickle_module=TRUSTED_LEGACY_PICKLE,
        )


def convert_checkpoint(source: Path, destination: Path) -> int:
    source = Path(source)
    destination = Path(destination)
    checkpoint = _load_checkpoint(source)
    state = checkpoint.get("state_dict", checkpoint)
    generator_state = {
        key[len("generator."):]: value
        for key, value in state.items()
        if key.startswith("generator.") and isinstance(value, torch.Tensor)
    }
    if not generator_state:
        raise ValueError(f"No generator tensors found in checkpoint: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(generator_state, destination)
    verified = torch.load(
        destination,
        map_location="cpu",
        weights_only=True,
    )
    if set(verified) != set(generator_state):
        raise RuntimeError("Converted generator artifact failed verification")
    return len(generator_state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    count = convert_checkpoint(args.source, args.destination)
    print(f"Converted {count} generator tensors to {args.destination}")


if __name__ == "__main__":
    main()
