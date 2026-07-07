"""Bruit blanc gaussien reproductible."""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ..common import _random_value
from .base import TypedSignal

class GaussianNoise(TypedSignal):
    signal_type = "gaussian_noise"
    allowed_windows = (None, "hann", "hamming", "boxcar")
    default_window = None

    STD_RANGE: ClassVar[tuple[float, float]] = (0.01, 0.1)
    DURATION_RANGE: ClassVar[tuple[float, float]] = (0.05, 1.0)
    RANDOM_SEED_MAX: ClassVar[int] = int(np.iinfo(np.uint32).max)

    def __init__(
        self,
        freq: float,
        data: np.ndarray,
        std: float,
        time_duration: float,
        seed: int,
    ):
        super().__init__(data=data, freq=freq)
        self.std = std
        self.time_duration = time_duration
        self.seed = seed

    @classmethod
    def generate(
        cls,
        freq: float,
        std: float,
        time_duration: float,
        seed: int,
    ) -> "GaussianNoise":
        n_samples = int(round(time_duration * freq))
        rng = np.random.default_rng(seed)
        data = rng.normal(0.0, std, n_samples)
        return cls(
            freq=freq,
            data=data,
            std=std,
            time_duration=time_duration,
            seed=int(seed),
        )

    @classmethod
    def generate_random_params(
        cls,
        rng: np.random.Generator,
        freq: float,
        fixed_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        fixed_params = fixed_params or {}
        seed = (
            int(fixed_params["seed"])
            if "seed" in fixed_params
            else int(rng.integers(0, cls.RANDOM_SEED_MAX, dtype=np.uint32))
        )
        return {
            "freq": freq,
            "std": _random_value(
                rng, fixed_params.get("std"), *cls.STD_RANGE
            ),
            "time_duration": _random_value(
                rng,
                fixed_params.get("time_duration"),
                *cls.DURATION_RANGE,
            ),
            "seed": seed,
        }


