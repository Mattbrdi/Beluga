"""Bruit sous-marin synthétique d'un grand navire à moteur."""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ..common import _random_value
from .base import TypedSignal


class LargeShipNoise(TypedSignal):
    """Combine cavitation, raies d'helice, raies moteur et modulation lente.

    Le bruit de cavitation possède un maximum spectral à basse fréquence puis
    décroît vers les hautes fréquences. Les raies sont placées aux multiples
    de la fréquence de passage des pales. Des raies distinctes représentent
    la cadence globale de combustion d'un diesel deux-temps lent et ses
    harmoniques transmis par la coque. Une modulation lente représente la
    périodicité du propulseur. Le signal final est normalisé à un pic unitaire;
    son niveau dans une scène est fixé par le gain du placement.
    """

    signal_type = "large_ship_noise"
    allowed_windows = (None,)
    default_window = None

    DURATION_RANGE: ClassVar[tuple[float, float]] = (2.0, 10.0)
    SHAFT_ROTATION_FREQUENCY_RANGE: ClassVar[tuple[float, float]] = (0.8, 2.0)
    PROPELLER_BLADE_COUNTS: ClassVar[tuple[int, ...]] = (4, 5, 6)
    TONAL_HARMONIC_COUNT_RANGE: ClassVar[tuple[int, int]] = (3, 7)
    TONAL_DECAY_RANGE: ClassVar[tuple[float, float]] = (0.8, 1.5)
    ENGINE_CYLINDER_COUNTS: ClassVar[tuple[int, ...]] = (6, 7, 8, 9, 10, 11, 12)
    ENGINE_HARMONIC_COUNT_RANGE: ClassVar[tuple[int, int]] = (5, 12)
    ENGINE_HARMONIC_DECAY_RANGE: ClassVar[tuple[float, float]] = (0.6, 1.3)
    ENGINE_MIX_RANGE: ClassVar[tuple[float, float]] = (0.08, 0.25)
    CAVITATION_PEAK_FREQUENCY_RANGE: ClassVar[tuple[float, float]] = (35.0, 90.0)
    LOW_FREQUENCY_SLOPE_RANGE: ClassVar[tuple[float, float]] = (1.0, 2.0)
    HIGH_FREQUENCY_SLOPE_RANGE: ClassVar[tuple[float, float]] = (0.8, 1.4)
    MODULATION_FREQUENCY_RANGE: ClassVar[tuple[float, float]] = (1.0, 4.0)
    MODULATION_DEPTH_RANGE: ClassVar[tuple[float, float]] = (0.1, 0.4)
    TONAL_MIX_RANGE: ClassVar[tuple[float, float]] = (0.15, 0.4)
    RANDOM_SEED_MAX: ClassVar[int] = int(np.iinfo(np.uint32).max)

    def __init__(
        self,
        freq: float,
        data: np.ndarray,
        time_duration: float,
        shaft_rotation_frequency: float,
        propeller_blade_count: int,
        tonal_harmonic_count: int,
        tonal_decay: float,
        engine_cylinder_count: int,
        engine_harmonic_count: int,
        engine_harmonic_decay: float,
        cavitation_peak_frequency: float,
        low_frequency_slope: float,
        high_frequency_slope: float,
        modulation_frequency: float,
        modulation_depth: float,
        tonal_mix: float,
        engine_mix: float,
        seed: int,
    ):
        super().__init__(data=data, freq=freq)
        self.time_duration = time_duration
        self.shaft_rotation_frequency = shaft_rotation_frequency
        self.propeller_blade_count = propeller_blade_count
        self.tonal_harmonic_count = tonal_harmonic_count
        self.tonal_decay = tonal_decay
        self.engine_cylinder_count = engine_cylinder_count
        self.engine_harmonic_count = engine_harmonic_count
        self.engine_harmonic_decay = engine_harmonic_decay
        self.cavitation_peak_frequency = cavitation_peak_frequency
        self.low_frequency_slope = low_frequency_slope
        self.high_frequency_slope = high_frequency_slope
        self.modulation_frequency = modulation_frequency
        self.modulation_depth = modulation_depth
        self.tonal_mix = tonal_mix
        self.engine_mix = engine_mix
        self.seed = seed

    @classmethod
    def generate(
        cls,
        freq: float,
        time_duration: float,
        shaft_rotation_frequency: float,
        propeller_blade_count: int,
        tonal_harmonic_count: int,
        tonal_decay: float,
        engine_cylinder_count: int,
        engine_harmonic_count: int,
        engine_harmonic_decay: float,
        cavitation_peak_frequency: float,
        low_frequency_slope: float,
        high_frequency_slope: float,
        modulation_frequency: float,
        modulation_depth: float,
        tonal_mix: float,
        engine_mix: float,
        seed: int,
    ) -> "LargeShipNoise":
        cls._validate_parameters(
            freq=freq,
            time_duration=time_duration,
            shaft_rotation_frequency=shaft_rotation_frequency,
            propeller_blade_count=propeller_blade_count,
            tonal_harmonic_count=tonal_harmonic_count,
            tonal_decay=tonal_decay,
            engine_cylinder_count=engine_cylinder_count,
            engine_harmonic_count=engine_harmonic_count,
            engine_harmonic_decay=engine_harmonic_decay,
            cavitation_peak_frequency=cavitation_peak_frequency,
            low_frequency_slope=low_frequency_slope,
            high_frequency_slope=high_frequency_slope,
            modulation_frequency=modulation_frequency,
            modulation_depth=modulation_depth,
            tonal_mix=tonal_mix,
            engine_mix=engine_mix,
        )
        rng = np.random.default_rng(seed)
        n_samples = int(round(freq * time_duration))
        time = np.arange(n_samples) / freq

        broadband = cls._colored_cavitation_noise(
            rng=rng,
            n_samples=n_samples,
            freq=freq,
            peak_frequency=cavitation_peak_frequency,
            low_slope=low_frequency_slope,
            high_slope=high_frequency_slope,
        )
        modulation_phase = rng.uniform(0.0, 2.0 * np.pi)
        modulation = 1.0 + modulation_depth * np.sin(
            2.0 * np.pi * modulation_frequency * time + modulation_phase
        )
        broadband *= modulation
        broadband = cls._normalize_rms(broadband)

        blade_rate = shaft_rotation_frequency * propeller_blade_count
        propeller_tones = cls._harmonic_tones(
            rng, time, blade_rate, tonal_harmonic_count, tonal_decay
        )
        engine_firing_frequency = shaft_rotation_frequency * engine_cylinder_count
        engine_tones = cls._harmonic_tones(
            rng,
            time,
            engine_firing_frequency,
            engine_harmonic_count,
            engine_harmonic_decay,
        )

        broadband_mix = 1.0 - tonal_mix - engine_mix
        data = (
            broadband_mix * broadband
            + tonal_mix * propeller_tones
            + engine_mix * engine_tones
        )
        peak = float(np.max(np.abs(data)))
        if peak > 0:
            data /= peak

        return cls(
            freq=freq,
            data=data,
            time_duration=time_duration,
            shaft_rotation_frequency=shaft_rotation_frequency,
            propeller_blade_count=int(propeller_blade_count),
            tonal_harmonic_count=int(tonal_harmonic_count),
            tonal_decay=tonal_decay,
            engine_cylinder_count=int(engine_cylinder_count),
            engine_harmonic_count=int(engine_harmonic_count),
            engine_harmonic_decay=engine_harmonic_decay,
            cavitation_peak_frequency=cavitation_peak_frequency,
            low_frequency_slope=low_frequency_slope,
            high_frequency_slope=high_frequency_slope,
            modulation_frequency=modulation_frequency,
            modulation_depth=modulation_depth,
            tonal_mix=tonal_mix,
            engine_mix=engine_mix,
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
        count_min, count_max = cls.TONAL_HARMONIC_COUNT_RANGE
        tonal_harmonic_count = (
            int(fixed_params["tonal_harmonic_count"])
            if "tonal_harmonic_count" in fixed_params
            else int(rng.integers(count_min, count_max + 1))
        )
        propeller_blade_count = (
            int(fixed_params["propeller_blade_count"])
            if "propeller_blade_count" in fixed_params
            else int(rng.choice(cls.PROPELLER_BLADE_COUNTS))
        )
        engine_cylinder_count = (
            int(fixed_params["engine_cylinder_count"])
            if "engine_cylinder_count" in fixed_params
            else int(rng.choice(cls.ENGINE_CYLINDER_COUNTS))
        )
        engine_count_min, engine_count_max = cls.ENGINE_HARMONIC_COUNT_RANGE
        engine_harmonic_count = (
            int(fixed_params["engine_harmonic_count"])
            if "engine_harmonic_count" in fixed_params
            else int(rng.integers(engine_count_min, engine_count_max + 1))
        )
        seed = (
            int(fixed_params["seed"])
            if "seed" in fixed_params
            else int(rng.integers(0, cls.RANDOM_SEED_MAX, dtype=np.uint32))
        )
        return {
            "freq": freq,
            "time_duration": _random_value(
                rng, fixed_params.get("time_duration"), *cls.DURATION_RANGE
            ),
            "shaft_rotation_frequency": _random_value(
                rng,
                fixed_params.get("shaft_rotation_frequency"),
                *cls.SHAFT_ROTATION_FREQUENCY_RANGE,
            ),
            "propeller_blade_count": propeller_blade_count,
            "tonal_harmonic_count": tonal_harmonic_count,
            "tonal_decay": _random_value(
                rng, fixed_params.get("tonal_decay"), *cls.TONAL_DECAY_RANGE
            ),
            "engine_cylinder_count": engine_cylinder_count,
            "engine_harmonic_count": engine_harmonic_count,
            "engine_harmonic_decay": _random_value(
                rng,
                fixed_params.get("engine_harmonic_decay"),
                *cls.ENGINE_HARMONIC_DECAY_RANGE,
            ),
            "cavitation_peak_frequency": _random_value(
                rng,
                fixed_params.get("cavitation_peak_frequency"),
                *cls.CAVITATION_PEAK_FREQUENCY_RANGE,
            ),
            "low_frequency_slope": _random_value(
                rng,
                fixed_params.get("low_frequency_slope"),
                *cls.LOW_FREQUENCY_SLOPE_RANGE,
            ),
            "high_frequency_slope": _random_value(
                rng,
                fixed_params.get("high_frequency_slope"),
                *cls.HIGH_FREQUENCY_SLOPE_RANGE,
            ),
            "modulation_frequency": _random_value(
                rng,
                fixed_params.get("modulation_frequency"),
                *cls.MODULATION_FREQUENCY_RANGE,
            ),
            "modulation_depth": _random_value(
                rng,
                fixed_params.get("modulation_depth"),
                *cls.MODULATION_DEPTH_RANGE,
            ),
            "tonal_mix": _random_value(
                rng, fixed_params.get("tonal_mix"), *cls.TONAL_MIX_RANGE
            ),
            "engine_mix": _random_value(
                rng, fixed_params.get("engine_mix"), *cls.ENGINE_MIX_RANGE
            ),
            "seed": seed,
        }

    @classmethod
    def _harmonic_tones(
        cls,
        rng: np.random.Generator,
        time: np.ndarray,
        fundamental_frequency: float,
        harmonic_count: int,
        harmonic_decay: float,
    ) -> np.ndarray:
        tones = np.zeros_like(time)
        for harmonic in range(1, harmonic_count + 1):
            amplitude = harmonic ** (-harmonic_decay)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            tones += amplitude * np.sin(
                2.0 * np.pi * harmonic * fundamental_frequency * time + phase
            )
        return cls._normalize_rms(tones)

    @staticmethod
    def _colored_cavitation_noise(
        rng: np.random.Generator,
        n_samples: int,
        freq: float,
        peak_frequency: float,
        low_slope: float,
        high_slope: float,
    ) -> np.ndarray:
        frequencies = np.fft.rfftfreq(n_samples, d=1.0 / freq)
        ratio = frequencies / peak_frequency
        power_shape = np.zeros_like(frequencies)
        positive = frequencies > 0
        positive_ratio = ratio[positive]
        power_shape[positive] = (
            positive_ratio**low_slope
            / (1.0 + positive_ratio ** (low_slope + high_slope))
        )
        random_spectrum = (
            rng.normal(size=len(frequencies))
            + 1j * rng.normal(size=len(frequencies))
        )
        random_spectrum *= np.sqrt(power_shape)
        random_spectrum[0] = 0.0
        return np.fft.irfft(random_spectrum, n=n_samples)

    @staticmethod
    def _normalize_rms(data: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(np.square(data))))
        return data if rms == 0 else data / rms

    @staticmethod
    def _validate_parameters(
        freq: float,
        time_duration: float,
        shaft_rotation_frequency: float,
        propeller_blade_count: int,
        tonal_harmonic_count: int,
        tonal_decay: float,
        engine_cylinder_count: int,
        engine_harmonic_count: int,
        engine_harmonic_decay: float,
        cavitation_peak_frequency: float,
        low_frequency_slope: float,
        high_frequency_slope: float,
        modulation_frequency: float,
        modulation_depth: float,
        tonal_mix: float,
        engine_mix: float,
    ) -> None:
        values = (
            freq,
            time_duration,
            shaft_rotation_frequency,
            tonal_decay,
            engine_harmonic_decay,
            cavitation_peak_frequency,
            low_frequency_slope,
            high_frequency_slope,
            modulation_frequency,
            modulation_depth,
            tonal_mix,
            engine_mix,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Tous les parametres doivent etre finis.")
        if freq <= 0 or time_duration <= 0:
            raise ValueError("freq et time_duration doivent etre strictement positifs.")
        if shaft_rotation_frequency <= 0 or cavitation_peak_frequency <= 0:
            raise ValueError("Les frequences caracteristiques doivent etre positives.")
        if (
            isinstance(tonal_harmonic_count, bool)
            or not isinstance(tonal_harmonic_count, (int, np.integer))
            or tonal_harmonic_count < 1
            or tonal_decay <= 0
        ):
            raise ValueError("La structure tonale doit contenir des harmoniques decroissantes.")
        for name, value in (
            ("propeller_blade_count", propeller_blade_count),
            ("engine_cylinder_count", engine_cylinder_count),
            ("engine_harmonic_count", engine_harmonic_count),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} doit etre un entier strictement positif.")
        if engine_harmonic_decay <= 0:
            raise ValueError("Les parametres du moteur doivent etre strictement positifs.")
        if low_frequency_slope <= 0 or high_frequency_slope <= 0:
            raise ValueError("Les pentes spectrales doivent etre strictement positives.")
        if modulation_frequency <= 0 or not 0 <= modulation_depth < 1:
            raise ValueError("Parametres de modulation invalides.")
        if not 0 <= tonal_mix <= 1 or not 0 <= engine_mix <= 1:
            raise ValueError("Les proportions du melange doivent etre comprises entre 0 et 1.")
        if tonal_mix + engine_mix > 1:
            raise ValueError("tonal_mix + engine_mix ne doit pas depasser 1.")
        blade_rate = shaft_rotation_frequency * propeller_blade_count
        if max(blade_rate, cavitation_peak_frequency) >= freq / 2.0:
            raise ValueError("Les frequences caracteristiques doivent rester sous Nyquist.")
        if tonal_harmonic_count * blade_rate >= freq / 2.0:
            raise ValueError("La plus haute raie de propulsion depasse Nyquist.")
        engine_firing_frequency = shaft_rotation_frequency * engine_cylinder_count
        if engine_harmonic_count * engine_firing_frequency >= freq / 2.0:
            raise ValueError("La plus haute raie moteur depasse Nyquist.")

    @property
    def blade_rate(self) -> float:
        return self.shaft_rotation_frequency * self.propeller_blade_count

    @property
    def engine_firing_frequency(self) -> float:
        return self.shaft_rotation_frequency * self.engine_cylinder_count
