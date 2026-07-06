"""Contrat commun des signaux mono générables aléatoirement."""
from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
from typing import Any

import numpy as np

from ...signal_class import Signal
from ..common import WindowName


class TypedSignal(Signal, ABC):
    signal_type: str = "generic"
    allowed_windows: tuple[WindowName, ...] = (None, "hann")
    default_window: WindowName = "hann"

    def __init__(self, data: np.ndarray, freq: float):
        super().__init__(data, freq)

    @classmethod
    def validate_generation_contract(cls) -> None:
        required_params = {"freq", "time_duration"}
        declared_params = set(inspect.signature(cls.generate).parameters)
        missing_params = required_params - declared_params
        if missing_params:
            raise TypeError(
                f"{cls.__name__}.generate() doit declarer les parametres "
                f"obligatoires {sorted(required_params)}; manquants: "
                f"{sorted(missing_params)}."
            )

    @classmethod
    def validate_fixed_params(cls, fixed_params: dict[str, Any] | None) -> None:
        if not fixed_params:
            return
        signature = inspect.signature(cls.generate)
        if any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            return
        accepted_params = set(signature.parameters)
        accepted_params.discard("freq")
        unknown_params = set(fixed_params) - accepted_params
        if unknown_params:
            raise ValueError(
                f"Parametres inconnus pour {cls.signal_type}: "
                f"{sorted(unknown_params)}. Parametres acceptes: "
                f"{sorted(accepted_params)}."
            )

    @classmethod
    @abstractmethod
    def generate(
        cls, freq: float, time_duration: float, **params: Any
    ) -> "TypedSignal":
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def generate_random_params(
        cls,
        rng: np.random.Generator,
        freq: float,
        fixed_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def generate_random(
        cls,
        rng: np.random.Generator,
        freq: float,
        fixed_params: dict[str, Any] | None = None,
    ) -> "TypedSignal":
        fixed_params = dict(fixed_params or {})
        cls.validate_fixed_params(fixed_params)
        params = cls.generate_random_params(
            rng=rng, freq=freq, fixed_params=fixed_params
        )
        return cls.generate(**params)

