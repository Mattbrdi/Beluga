"""Signal sinusoïdal."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..common import _random_value
from .base import TypedSignal

class SinSignal(TypedSignal):
    signal_type = "sine"
    allowed_windows = (None, "hann", "hamming", "blackman", "boxcar")
    default_window = "hann"

    def __init__(
        self,
        freq: float,
        data: np.ndarray,
        sin_freq: float,
        phase: float,
        amplitude: float,
        time_duration: float,
    ):
        super().__init__(data=data, freq=freq)
        self.sin_freq = sin_freq
        self.phase = phase
        self.amplitude = amplitude
        self.time_duration = time_duration

    @classmethod
    def generate(
        cls,
        freq: float,
        sin_freq: float,
        phase: float,
        amplitude: float,
        time_duration: float,
    ) -> "SinSignal":
        n_samples = int(round(time_duration * freq))
        time = np.arange(n_samples) / freq
        data = amplitude * np.sin(2 * np.pi * sin_freq * time + phase)
        return cls(
            freq=freq,
            data=data,
            sin_freq=sin_freq,
            phase=phase,
            amplitude=amplitude,
            time_duration=time_duration,
        )

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
            "sin_freq": _random_value(rng, fixed_params.get("sin_freq"), 100.0, min(3000.0, 0.45 * freq)),
            "phase": _random_value(rng, fixed_params.get("phase"), 0.0, 2 * np.pi),
            # Par defaut, les briques sont normalisees; le niveau de scene vient du placement.gain.
            "amplitude": _random_value(rng, fixed_params.get("amplitude"), 1.0, 1.0),
            "time_duration": _random_value(
                rng,
                fixed_params.get("time_duration"),
                0.05,
                1.0,
            ),
        }



