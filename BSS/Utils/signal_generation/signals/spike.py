"""Impulsion brève."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..common import _random_value
from .base import TypedSignal

class SpikeSignal(TypedSignal):
    signal_type = "spike"
    allowed_windows = (None, "boxcar", "hann")
    default_window = None

    def __init__(self, freq: float, data: np.ndarray, amplitude: float, time_duration: float):
        super().__init__(data=data, freq=freq)
        self.amplitude = amplitude
        self.time_duration = time_duration

    @classmethod
    def generate(cls, freq: float, amplitude: float, time_duration: float) -> "SpikeSignal":
        n_samples = max(1, int(round(time_duration * freq)))
        data = np.zeros(n_samples)
        data[0] = amplitude
        return cls(freq=freq, data=data, amplitude=amplitude, time_duration=time_duration)

    @classmethod
    def generate_random_params(
        cls,
        rng: np.random.Generator,
        freq: float,
        fixed_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        fixed_params = fixed_params or {}
        return {
            "freq": freq,
            # Par defaut, les briques sont normalisees; le niveau de scene vient du placement.gain.
            "amplitude": _random_value(rng, fixed_params.get("amplitude"), 1.0, 1.0),
            "time_duration": _random_value(
                rng,
                fixed_params.get("time_duration"),
                1.0 / freq,
                0.02,
            ),
        }



