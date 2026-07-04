"""Types partages et utilitaires de tirage aleatoire."""
from __future__ import annotations

from typing import Any

import numpy as np

WindowName = str | None
WindowChoice = WindowName | str
SignalTypeChoice = str | list[str] | tuple[str, ...] | None
RANDOM_WINDOW = "random"


def _random_value(
    rng: np.random.Generator,
    value: Any | None,
    low: float,
    high: float,
) -> float:
    if value is not None:
        return float(value)
    return float(rng.uniform(low, high))


def _random_choice(
    rng: np.random.Generator,
    value: Any | None,
    choices: list[Any] | tuple[Any, ...],
) -> Any:
    if value is not None:
        return value
    if not choices:
        raise ValueError("Impossible de choisir aleatoirement dans une liste vide.")
    return choices[int(rng.integers(0, len(choices)))]


def _random_type_name(
    rng: np.random.Generator,
    type_name: SignalTypeChoice, # None -> random in registry; list/tuple -> random in subset.
    registry: dict[str, type["TypedSignal"]],
) -> str:
    if not registry:
        raise ValueError("Aucun type de signal enregistre dans le registry demande.")

    if type_name is None:
        candidates = list(registry.keys())
    elif isinstance(type_name, str):
        candidates = [type_name]
    else:
        candidates = list(type_name)
        if not candidates:
            raise ValueError("La liste de types de signal autorises ne peut pas etre vide.")

    for candidate in candidates:
        if candidate not in registry:
            raise ValueError(f"Type de signal inconnu: {candidate}")

    return candidates[int(rng.integers(0, len(candidates)))]
