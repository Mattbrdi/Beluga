"""Outils de construction et de lecture des datasets synthetiques BSS."""

from .builder import build_dataset
from .config import DatasetConfig, GeneratorConfig
from .io import load_scene, load_scene_arrays
from .scenarios import SCENARIO_FACTORIES, register_scenario

__all__ = [
    "DatasetConfig",
    "GeneratorConfig",
    "SCENARIO_FACTORIES",
    "build_dataset",
    "load_scene",
    "load_scene_arrays",
    "register_scenario",
]
