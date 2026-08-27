"""Sifflement harmonique à contour fréquentiel lisse."""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import lfilter

from ..common import _random_value
from .base import TypedSignal

class WhistleSignal(TypedSignal):
    """Sifflement harmonique a contour frequentiel aleatoire et lisse.

    La generation construit quelques points de controle en faisant evoluer la
    frequence vers le haut ou le bas. Une interpolation PCHIP relie ces points
    sans oscillations artificielles. Un faible jitter temporellement correle
    est ensuite ajoute, puis la frequence instantanee est integree pour obtenir
    la phase. Le signal final somme les harmoniques demandees, applique une
    enveloppe lente et normalise le pic a 1. Le fenetrage temporel est laisse
    a ``SignalPlacement``.

    Parametres
    ----------
    freq:
        Frequence d'echantillonnage en Hz.
    time_duration:
        Duree totale du sifflement en secondes.
    f_start:
        Frequence fondamentale au debut du signal, en Hz.
    f_min, f_max:
        Bornes entre lesquelles la frequence fondamentale est contrainte.
    segment_duration_range:
        Durees minimale et maximale, en secondes, entre deux points de controle
        du contour frequentiel. Des segments longs produisent moins
        d'inflexions.
    direction_change_probability:
        Probabilite de changer spontanement le sens du balayage a chaque
        segment. Une borne f_min ou f_max force aussi un changement de sens.
    sweep_rate_range:
        Vitesses minimale et maximale de variation de la fondamentale, en Hz/s.
    jitter_tau:
        Constante de temps, en secondes, du filtre appliquant le jitter. Une
        valeur elevee produit des irregularites plus lentes.
    jitter_std:
        Ecart-type du jitter frequentiel, en Hz. Zero le desactive.
    harmonic_amplitudes:
        Amplitude de chaque harmonique, en commencant par la fondamentale. Le
        nombre d'elements definit le nombre total d'harmoniques.
    harmonic_phases:
        Decalage de phase, en radians, de chaque harmonique. Ce tuple doit avoir
        la meme longueur que harmonic_amplitudes.
    envelope_base:
        Niveau constant de l'enveloppe d'amplitude avant normalisation.
    envelope_depth:
        Amplitude de la variation lente ajoutee a envelope_base.
    seed:
        Graine controlant les points du contour, les changements de direction
        et le jitter. Les memes parametres et la meme graine reproduisent le
        meme signal.

    La derniere harmonique doit rester sous la frequence de Nyquist:
    ``len(harmonic_amplitudes) * f_max < freq / 2``.

    Par defaut, la generation aleatoire tire entre deux et quatre harmoniques.
    Leurs amplitudes suivent une decroissance en puissance perturbee par une
    faible variation log-normale, puis sont normalisees relativement a la
    fondamentale. Leurs phases sont tirees uniformement sur un tour complet.
    """

    signal_type = "whistle"
    allowed_windows = ("hann",)
    default_window = "hann"

    DURATION_RANGE: ClassVar[tuple[float, float]] = (0.35, 1.2)
    HARMONIC_COUNT_RANGE: ClassVar[tuple[int, int]] = (2, 4)
    HARMONIC_DECAY_RANGE: ClassVar[tuple[float, float]] = (1.5, 3.0)
    HARMONIC_VARIATION_STD: ClassVar[float] = 0.2
    HARMONIC_PHASE_RANGE: ClassVar[tuple[float, float]] = (0.0, 2.0 * np.pi)
    NYQUIST_MARGIN: ClassVar[float] = 0.45
    F_MAX_DEFAULT: ClassVar[float] = 4_000.0
    F_MIN_DEFAULT: ClassVar[float] = 800.0
    F_MIN_FRACTION_OF_MAX: ClassVar[float] = 0.5
    F_START_DEFAULT: ClassVar[float] = 2_000.0
    SEGMENT_DURATION_RANGE: ClassVar[tuple[float, float]] = (0.35, 0.9)
    DIRECTION_CHANGE_PROBABILITY: ClassVar[float] = 0.2
    SWEEP_RATE_RANGE: ClassVar[tuple[float, float]] = (300.0, 1_000.0)
    JITTER_TAU: ClassVar[float] = 0.05
    JITTER_STD: ClassVar[float] = 4.0
    ENVELOPE_BASE: ClassVar[float] = 0.75
    ENVELOPE_DEPTH: ClassVar[float] = 0.15
    RANDOM_SEED_MAX: ClassVar[int] = int(np.iinfo(np.uint32).max)

    def __init__(
        self,
        freq: float,
        data: np.ndarray,
        time_duration: float,
        f_start: float,
        f_min: float,
        f_max: float,
        segment_duration_range: tuple[float, float],
        direction_change_probability: float,
        sweep_rate_range: tuple[float, float],
        jitter_tau: float,
        jitter_std: float,
        harmonic_amplitudes: tuple[float, ...],
        harmonic_phases: tuple[float, ...],
        envelope_base: float,
        envelope_depth: float,
        seed: int,
    ):
        super().__init__(data=data, freq=freq)
        self.time_duration = time_duration
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.segment_duration_range = segment_duration_range
        self.direction_change_probability = direction_change_probability
        self.sweep_rate_range = sweep_rate_range
        self.jitter_tau = jitter_tau
        self.jitter_std = jitter_std
        self.harmonic_amplitudes = harmonic_amplitudes
        self.harmonic_phases = harmonic_phases
        self.envelope_base = envelope_base
        self.envelope_depth = envelope_depth
        self.seed = seed

    @classmethod
    def generate(
        cls,
        freq: float,
        time_duration: float,
        f_start: float,
        f_min: float,
        f_max: float,
        segment_duration_range: tuple[float, float],
        direction_change_probability: float,
        sweep_rate_range: tuple[float, float],
        jitter_tau: float,
        jitter_std: float,
        harmonic_amplitudes: tuple[float, ...],
        harmonic_phases: tuple[float, ...],
        envelope_base: float,
        envelope_depth: float,
        seed: int,
    ) -> "WhistleSignal":
        cls._validate_parameters(
            freq=freq,
            time_duration=time_duration,
            f_start=f_start,
            f_min=f_min,
            f_max=f_max,
            segment_duration_range=segment_duration_range,
            direction_change_probability=direction_change_probability,
            sweep_rate_range=sweep_rate_range,
            jitter_tau=jitter_tau,
            jitter_std=jitter_std,
            harmonic_amplitudes=harmonic_amplitudes,
            harmonic_phases=harmonic_phases,
        )

        rng = np.random.default_rng(seed)
        n_samples = int(round(freq * time_duration))
        time = np.arange(n_samples) / freq
        control_times, control_freqs = cls._generate_frequency_control_points(
            rng=rng,
            time_duration=time_duration,
            f_start=f_start,
            f_min=f_min,
            f_max=f_max,
            segment_duration_range=segment_duration_range,
            direction_change_probability=direction_change_probability,
            sweep_rate_range=sweep_rate_range,
        )

        contour = PchipInterpolator(control_times, control_freqs)(time)
        smoothing = np.exp(-1.0 / (freq * jitter_tau))
        jitter = lfilter(
            [jitter_std * np.sqrt(1.0 - smoothing**2)],
            [1.0, -smoothing],
            rng.normal(size=n_samples),
        )
        instantaneous_frequency = np.clip(contour + jitter, f_min, f_max)
        phase = 2.0 * np.pi * np.cumsum(instantaneous_frequency) / freq

        data = np.zeros_like(phase)
        for harmonic, (amplitude, phase_offset) in enumerate(
            zip(harmonic_amplitudes, harmonic_phases),
            start=1,
        ):
            data += amplitude * np.sin(harmonic * phase + phase_offset)
        envelope = envelope_base + envelope_depth * np.sin(
            np.pi * time / time_duration
        )
        data *= envelope
        peak = np.max(np.abs(data))
        if peak > 0:
            data /= peak

        return cls(
            freq=freq,
            data=data,
            time_duration=time_duration,
            f_start=f_start,
            f_min=f_min,
            f_max=f_max,
            segment_duration_range=(
                float(segment_duration_range[0]),
                float(segment_duration_range[1]),
            ),
            direction_change_probability=direction_change_probability,
            sweep_rate_range=(
                float(sweep_rate_range[0]),
                float(sweep_rate_range[1]),
            ),
            jitter_tau=jitter_tau,
            jitter_std=jitter_std,
            harmonic_amplitudes=tuple(harmonic_amplitudes),
            harmonic_phases=tuple(harmonic_phases),
            envelope_base=envelope_base,
            envelope_depth=envelope_depth,
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
        harmonic_amplitudes, harmonic_phases = cls._random_harmonic_structure(
            rng=rng,
            fixed_amplitudes=fixed_params.get("harmonic_amplitudes"),
            fixed_phases=fixed_params.get("harmonic_phases"),
        )
        n_harmonics = max(1, len(harmonic_amplitudes))
        max_fundamental = cls.NYQUIST_MARGIN * freq / n_harmonics
        f_max = float(
            fixed_params.get("f_max", min(cls.F_MAX_DEFAULT, max_fundamental))
        )
        f_min = float(
            fixed_params.get(
                "f_min", min(cls.F_MIN_DEFAULT, cls.F_MIN_FRACTION_OF_MAX * f_max)
            )
        )
        f_start = float(
            fixed_params.get(
                "f_start", np.clip(cls.F_START_DEFAULT, f_min, f_max)
            )
        )
        seed = (
            int(fixed_params["seed"])
            if "seed" in fixed_params
            else int(rng.integers(0, cls.RANDOM_SEED_MAX, dtype=np.uint32))
        )

        return {
            "freq": freq,
            "time_duration": _random_value(
                rng,
                fixed_params.get("time_duration"),
                *cls.DURATION_RANGE,
            ),
            "f_start": f_start,
            "f_min": f_min,
            "f_max": f_max,
            "segment_duration_range": tuple(
                fixed_params.get("segment_duration_range", cls.SEGMENT_DURATION_RANGE)
            ),
            "direction_change_probability": float(
                fixed_params.get(
                    "direction_change_probability", cls.DIRECTION_CHANGE_PROBABILITY
                )
            ),
            "sweep_rate_range": tuple(
                fixed_params.get("sweep_rate_range", cls.SWEEP_RATE_RANGE)
            ),
            "jitter_tau": float(fixed_params.get("jitter_tau", cls.JITTER_TAU)),
            "jitter_std": float(fixed_params.get("jitter_std", cls.JITTER_STD)),
            "harmonic_amplitudes": harmonic_amplitudes,
            "harmonic_phases": harmonic_phases,
            "envelope_base": float(
                fixed_params.get("envelope_base", cls.ENVELOPE_BASE)
            ),
            "envelope_depth": float(
                fixed_params.get("envelope_depth", cls.ENVELOPE_DEPTH)
            ),
            "seed": seed,
        }

    @classmethod
    def _random_harmonic_structure(
        cls,
        rng: np.random.Generator,
        fixed_amplitudes: tuple[float, ...] | list[float] | None,
        fixed_phases: tuple[float, ...] | list[float] | None,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Tire un spectre décroissant, sauf valeurs explicitement fixées."""
        if fixed_amplitudes is not None:
            harmonic_count = len(fixed_amplitudes)
        elif fixed_phases is not None:
            harmonic_count = len(fixed_phases)
        else:
            count_min, count_max = cls.HARMONIC_COUNT_RANGE
            harmonic_count = int(rng.integers(count_min, count_max + 1))

        if fixed_amplitudes is None:
            decay = float(rng.uniform(*cls.HARMONIC_DECAY_RANGE))
            harmonic_orders = np.arange(1, harmonic_count + 1, dtype=float)
            variations = rng.lognormal(
                mean=0.0,
                sigma=cls.HARMONIC_VARIATION_STD,
                size=harmonic_count,
            )
            amplitudes = harmonic_orders ** (-decay) * variations
            if harmonic_count > 0:
                amplitudes /= amplitudes[0]
            harmonic_amplitudes = tuple(float(value) for value in amplitudes)
        else:
            harmonic_amplitudes = tuple(float(value) for value in fixed_amplitudes)

        if fixed_phases is None:
            phases = rng.uniform(
                *cls.HARMONIC_PHASE_RANGE,
                size=harmonic_count,
            )
            harmonic_phases = tuple(float(value) for value in phases)
        else:
            harmonic_phases = tuple(float(value) for value in fixed_phases)

        return harmonic_amplitudes, harmonic_phases

    @staticmethod
    def _generate_frequency_control_points(
        rng: np.random.Generator,
        time_duration: float,
        f_start: float,
        f_min: float,
        f_max: float,
        segment_duration_range: tuple[float, float],
        direction_change_probability: float,
        sweep_rate_range: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        control_times = [0.0]
        control_freqs = [f_start]
        direction = int(rng.choice([-1, 1]))

        while control_times[-1] < time_duration:
            segment_duration = rng.uniform(*segment_duration_range)
            next_time = min(control_times[-1] + segment_duration, time_duration)
            actual_duration = next_time - control_times[-1]

            if rng.random() < direction_change_probability:
                direction *= -1

            sweep_rate = rng.uniform(*sweep_rate_range)
            next_frequency = control_freqs[-1] + direction * sweep_rate * actual_duration
            if next_frequency > f_max:
                next_frequency = f_max
                direction = -1
            elif next_frequency < f_min:
                next_frequency = f_min
                direction = 1

            control_times.append(next_time)
            control_freqs.append(next_frequency)

        return np.asarray(control_times), np.asarray(control_freqs)

    @staticmethod
    def _validate_parameters(
        freq: float,
        time_duration: float,
        f_start: float,
        f_min: float,
        f_max: float,
        segment_duration_range: tuple[float, float],
        direction_change_probability: float,
        sweep_rate_range: tuple[float, float],
        jitter_tau: float,
        jitter_std: float,
        harmonic_amplitudes: tuple[float, ...],
        harmonic_phases: tuple[float, ...],
    ) -> None:
        if freq <= 0 or time_duration <= 0:
            raise ValueError("freq et time_duration doivent etre strictement positifs.")
        if int(round(freq * time_duration)) < 1:
            raise ValueError("time_duration est trop courte pour produire un echantillon.")
        if not 0 < f_min <= f_start <= f_max:
            raise ValueError("Les frequences doivent verifier 0 < f_min <= f_start <= f_max.")
        if len(segment_duration_range) != 2 or not (
            0 < segment_duration_range[0] <= segment_duration_range[1]
        ):
            raise ValueError("segment_duration_range doit contenir deux durees positives ordonnees.")
        if len(sweep_rate_range) != 2 or not (
            0 <= sweep_rate_range[0] <= sweep_rate_range[1]
        ):
            raise ValueError("sweep_rate_range doit contenir deux vitesses positives ordonnees.")
        if not 0 <= direction_change_probability <= 1:
            raise ValueError("direction_change_probability doit etre comprise entre 0 et 1.")
        if jitter_tau <= 0 or jitter_std < 0:
            raise ValueError("Les parametres de jitter doivent etre positifs.")
        if not harmonic_amplitudes:
            raise ValueError("harmonic_amplitudes doit contenir au moins une harmonique.")
        if len(harmonic_amplitudes) != len(harmonic_phases):
            raise ValueError(
                "harmonic_amplitudes et harmonic_phases doivent avoir la meme longueur."
            )
        if not all(np.isfinite(value) for value in (*harmonic_amplitudes, *harmonic_phases)):
            raise ValueError("Les amplitudes et phases harmoniques doivent etre finies.")
        if not any(amplitude != 0 for amplitude in harmonic_amplitudes):
            raise ValueError("Au moins une amplitude harmonique doit etre non nulle.")

        highest_harmonic = len(harmonic_amplitudes)
        if highest_harmonic * f_max >= freq / 2.0:
            raise ValueError(
                "La plus haute harmonique depasse Nyquist; augmente freq ou reduis f_max."
            )
