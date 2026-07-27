"""Model adapters with lazy access to the global manager.

Importing a concrete adapter must not eagerly import every optional model
dependency (SAM3, diffusers, HD-Painter, and others).
"""

__all__ = ["model_manager"]


def __getattr__(name):
    if name != "model_manager":
        raise AttributeError(name)
    from .manager import model_manager

    return model_manager

__all__ = ["model_manager"]
