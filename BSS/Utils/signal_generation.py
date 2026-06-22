from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import signal as sp_signal
from collections.abc import Sequence

from BSS.Utils.signal_class import Signal, MultiSignal, Mixture

"""
Generateur de scenes acoustiques synthetiques pour tester la separation de sources.

Principe:
- un TypedSignal est une brique temporelle courte ou continue;
- un SignalPlacement place une brique dans le temps avec une fenetre et un gain;
- un CompositeSignal additionne plusieurs placements pour former une source mono;
- AudioSceneGenerator construit les sources, applique une Mixture source->micros,
  puis ajoute les bruits locaux et continus.
"""


WindowName = str | None
WindowChoice = WindowName | str
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


def _random_int(
    rng: np.random.Generator,
    value: int | None,
    low: int,
    high_inclusive: int,
) -> int:
    if value is not None:
        return int(value)
    return int(rng.integers(low, high_inclusive + 1))


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
    type_name: str | None,
    registry: dict[str, type["TypedSignal"]],
) -> str:
    if type_name is not None:
        if type_name not in registry:
            raise ValueError(f"Type de signal inconnu: {type_name}")
        return type_name
    if not registry:
        raise ValueError("Aucun type de signal enregistre dans le registry demande.")
    keys = list(registry.keys())
    return keys[int(rng.integers(0, len(keys)))]


class TypedSignal(Signal, ABC):
    signal_type: str = "generic"
    allowed_windows: tuple[WindowName, ...] = (None, "hann")
    default_window: WindowName = "hann"

    def __init__(self, data: np.ndarray, freq: float):
        super().__init__(data, freq)

    @classmethod
    @abstractmethod
    def generate(cls, **params: Any) -> "TypedSignal":
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def generate_random_params(
        cls,
        rng: np.random.Generator,
        freq: float,
        scene_duration: float,
        fixed_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def generate_random(
        cls,
        rng: np.random.Generator,
        freq: float,
        scene_duration: float,
        fixed_params: dict[str, Any] | None = None,
    ) -> "TypedSignal":
        params = cls.generate_random_params(
            rng=rng,
            freq=freq,
            scene_duration=scene_duration,
            fixed_params=fixed_params,
        )
        return cls.generate(**params)


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
        scene_duration: float,
        fixed_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        fixed_params = fixed_params or {}
        return {
            "freq": freq,
            "sin_freq": _random_value(rng, fixed_params.get("sin_freq"), 100.0, min(3000.0, 0.45 * freq)),
            "phase": _random_value(rng, fixed_params.get("phase"), 0.0, 2 * np.pi),
            "amplitude": _random_value(rng, fixed_params.get("amplitude"), 0.5, 1.0),
            "time_duration": _random_value(
                rng,
                fixed_params.get("time_duration"),
                0.05,
                max(0.05, min(1.0, scene_duration)),
            ),
        }


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
        scene_duration: float,
        fixed_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        fixed_params = fixed_params or {}
        return {
            "freq": freq,
            "amplitude": _random_value(rng, fixed_params.get("amplitude"), 0.5, 2.0),
            "time_duration": _random_value(
                rng,
                fixed_params.get("time_duration"),
                1.0 / freq,
                min(0.02, scene_duration),
            ),
        }


class GaussianNoise(TypedSignal):
    signal_type = "gaussian_noise"
    allowed_windows = (None, "hann", "hamming", "boxcar")
    default_window = None

    def __init__(self, freq: float, data: np.ndarray, std: float, time_duration: float):
        super().__init__(data=data, freq=freq)
        self.std = std
        self.time_duration = time_duration

    @classmethod
    def generate(cls, freq: float, std: float, time_duration: float) -> "GaussianNoise":
        n_samples = int(round(time_duration * freq))
        data = np.random.default_rng().normal(0.0, std, n_samples)
        return cls(freq=freq, data=data, std=std, time_duration=time_duration)

    @classmethod
    def generate_with_rng(
        cls,
        rng: np.random.Generator,
        freq: float,
        std: float,
        time_duration: float,
    ) -> "GaussianNoise":
        n_samples = int(round(time_duration * freq))
        data = rng.normal(0.0, std, n_samples)
        return cls(freq=freq, data=data, std=std, time_duration=time_duration)

    @classmethod
    def generate_random_params(
        cls,
        rng: np.random.Generator,
        freq: float,
        scene_duration: float,
        fixed_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        fixed_params = fixed_params or {}
        return {
            "freq": freq,
            "std": _random_value(rng, fixed_params.get("std"), 0.01, 0.1),
            "time_duration": _random_value(
                rng,
                fixed_params.get("time_duration"),
                0.05,
                max(0.05, min(1.0, scene_duration)),
            ),
        }

    @classmethod
    def generate_random(
        cls,
        rng: np.random.Generator,
        freq: float,
        scene_duration: float,
        fixed_params: dict[str, Any] | None = None,
    ) -> "GaussianNoise":
        params = cls.generate_random_params(rng, freq, scene_duration, fixed_params)
        return cls.generate_with_rng(rng=rng, **params)


@dataclass
class SignalPlacementSpec:
    """Contraintes partielles pour un placement: les champs None/random sont tires aleatoirement."""
    signal_type: str | None = None
    signal_params: dict[str, Any] = field(default_factory=dict)
    start_time: float | None = None
    window: WindowChoice = RANDOM_WINDOW
    gain: float | None = None


@dataclass
class CompositeSignalSpec:
    """Contraintes partielles pour une source composite et ses placements."""
    n_placements: int | None = None
    placements: list[SignalPlacementSpec] = field(default_factory=list)


@dataclass
class SignalPlacement:
    signal: TypedSignal
    start_time: float
    window: WindowName = None
    gain: float = 1.0

    def __post_init__(self) -> None:
        if self.window not in self.signal.allowed_windows:
            raise ValueError(
                f"Fenetre {self.window} non autorisee pour {self.signal.signal_type}."
            )

    def is_compatible_with(self, other: "SignalPlacement") -> bool:
        if not isinstance(other, SignalPlacement):
            raise TypeError("other doit etre une instance de SignalPlacement.")
        return self.signal.freq == other.signal.freq

    def render(self) -> Signal:
        data_array = self._apply_window(self.signal) * self.gain
        latency = Signal.from_zeros(duration=self.start_time, freq=self.signal.freq)
        return Signal.concat(latency, data_array)

    def _apply_window(self, signal: Signal) -> Signal:
        if self.window is None:
            return signal.copy()
        window = Signal(
            data=sp_signal.get_window(window=self.window, Nx=len(signal.data)),
            freq=signal.freq,
        )
        return signal * window


class CompositeSignal(TypedSignal):
    signal_type = "composite"
    allowed_windows = (None,)
    default_window = None

    def __init__(self, placements: list[SignalPlacement], freq: float | None = None):
        if not placements and freq is None:
            raise ValueError("Un CompositeSignal vide doit recevoir une frequence freq.")
        if placements and not self.verify_frequency_coherence(placements):
            raise ValueError("Tous les SignalPlacement doivent avoir la meme frequence.")

        self.placements = list(placements)
        if placements:
            render = self.render()
            super().__init__(data=render.data, freq=render.freq)
        else:
            if freq is not None:
                super().__init__(data=np.zeros(0), freq=float(freq)) 

    @classmethod
    def generate(cls, placements: list[SignalPlacement], freq: float | None = None) -> "CompositeSignal":
        return cls(placements=placements, freq=freq)

    @classmethod
    def generate_random_params(
        cls,
        rng: np.random.Generator,
        freq: float,
        scene_duration: float,
        fixed_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("CompositeSignal est genere par CompositeSignalGenerator.")

    @classmethod
    def verify_frequency_coherence(cls, placements: list[SignalPlacement]) -> bool:
        if len(placements) <= 1:
            return True
        reference = placements[0]
        return all(reference.is_compatible_with(place) for place in placements[1:])

    def render(self) -> Signal:
        if not self.placements:
            return Signal(np.zeros(0), self.freq)
        signal = self.placements[0].render()
        for placement in self.placements[1:]:
            signal = signal + placement.render()
        return signal

    def cut(self, start_time: float | None = None, end_time: float | None = None) -> Signal:
        signal = self.render()
        start_time = 0.0 if start_time is None else start_time
        end_time = signal.duration if end_time is None else end_time

        if start_time < 0 or end_time < 0:
            raise ValueError("start_time et end_time doivent etre positifs.")
        if end_time < start_time:
            raise ValueError("end_time doit etre superieur ou egal a start_time.")

        start_idx = int(round(start_time * signal.freq))
        end_idx = int(round(end_time * signal.freq))
        return Signal(signal.data[start_idx:end_idx].copy(), signal.freq)

    def cut_to_duration(self, duration: float) -> Signal:
        signal = self.cut(0.0, duration)
        target_samples = int(round(duration * self.freq))
        if len(signal.data) < target_samples:
            signal = Signal(
                np.pad(signal.data, (0, target_samples - len(signal.data))),
                signal.freq,
            )
        return signal

    def add_placement(self, placement: SignalPlacement) -> None:
        if self.placements and not placement.is_compatible_with(self.placements[0]):
            raise ValueError("SignalPlacement non compatible.")
        if not self.placements and len(self.data) == 0:
            self.freq = placement.signal.freq
        self.placements.append(placement)
        rendered = self.render()
        self.data = rendered.data


_SOURCE_SIGNAL_TYPE: dict[str, type[TypedSignal]] = {}
_LOCAL_NOISE_SIGNAL_TYPE: dict[str, type[TypedSignal]] = {}
_CONTINUOUS_NOISE_SIGNAL_TYPE: dict[str, type[TypedSignal]] = {}


def register_SourceSignal(signal_cls: type[TypedSignal]) -> None:
    if not issubclass(signal_cls, TypedSignal):
        raise TypeError("La classe doit heriter de TypedSignal.")
    _SOURCE_SIGNAL_TYPE[signal_cls.signal_type] = signal_cls


def register_LocalNoiseSignal(signal_cls: type[TypedSignal]) -> None:
    if not issubclass(signal_cls, TypedSignal):
        raise TypeError("La classe doit heriter de TypedSignal.")
    _LOCAL_NOISE_SIGNAL_TYPE[signal_cls.signal_type] = signal_cls


def register_ContinuousNoiseSignal(signal_cls: type[TypedSignal]) -> None:
    if not issubclass(signal_cls, TypedSignal):
        raise TypeError("La classe doit heriter de TypedSignal.")
    _CONTINUOUS_NOISE_SIGNAL_TYPE[signal_cls.signal_type] = signal_cls


class SignalGenerator:
    def __init__(self, registry: dict[str, type[TypedSignal]]):
        self.registry = registry

    def generate_placement(
        self,
        rng: np.random.Generator,
        freq: float,
        scene_duration: float,
        gain_range: tuple[float, float],
        spec: SignalPlacementSpec | None = None,
    ) -> SignalPlacement:
        spec = spec or SignalPlacementSpec()
        type_name = _random_type_name(rng, spec.signal_type, self.registry)
        signal_cls = self.registry[type_name]
        signal = signal_cls.generate_random(
            rng=rng,
            freq=freq,
            scene_duration=scene_duration,
            fixed_params=spec.signal_params,
        )
        max_start = max(0.0, scene_duration - signal.duration)
        start_time = _random_value(rng, spec.start_time, 0.0, max_start)
        window = (
            _random_choice(rng, None, signal_cls.allowed_windows)
            if spec.window == RANDOM_WINDOW
            else spec.window
        )
        gain = _random_value(rng, spec.gain, gain_range[0], gain_range[1])
        return SignalPlacement(
            signal=signal,
            start_time=start_time,
            window=window,
            gain=gain,
        )


class CompositeSignalGenerator:
    def __init__(
        self,
        registry: dict[str, type[TypedSignal]],
        n_placements_range: tuple[int, int],
        gain_range: tuple[float, float],
    ):
        self.signal_generator = SignalGenerator(registry)
        self.n_placements_range = n_placements_range
        self.gain_range = gain_range

    def generate(
        self,
        rng: np.random.Generator,
        freq: float,
        scene_duration: float,
        spec: CompositeSignalSpec | None = None,
    ) -> CompositeSignal:
        spec = spec or CompositeSignalSpec()
        n_placements = _random_int(
            rng,
            spec.n_placements,
            self.n_placements_range[0],
            self.n_placements_range[1],
        )

        placement_specs = list(spec.placements)
        if len(placement_specs) > n_placements:
            raise ValueError("Plus de placements fixes que n_placements.")
        while len(placement_specs) < n_placements:
            placement_specs.append(SignalPlacementSpec())

        placements = [
            self.signal_generator.generate_placement(
                rng=rng,
                freq=freq,
                scene_duration=scene_duration,
                gain_range=self.gain_range,
                spec=placement_spec,
            )
            for placement_spec in placement_specs
        ]
        composite = CompositeSignal(placements=placements, freq=freq)
        composite.data = composite.cut_to_duration(scene_duration).data
        return composite


class MixtureGenerator:
    def __init__(self, max_delay: int):
        if max_delay < 0:
            raise ValueError("max_delay doit etre positif ou nul.")
        self.max_delay = int(max_delay)

    def generate(
        self,
        rng: np.random.Generator,
        n_sources: int,
        n_mics: int,
        delay_matrix: np.ndarray | None = None,
    ) -> Mixture:
        if delay_matrix is None:
            delay_matrix = rng.integers(
                0,
                self.max_delay + 1,
                size=(n_mics, n_sources),
            )
        delay_matrix = np.asarray(delay_matrix, dtype=int)
        if delay_matrix.shape != (n_mics, n_sources):
            raise ValueError(
                f"delay_matrix doit etre de shape {(n_mics, n_sources)}, obtenu {delay_matrix.shape}."
            )
        return Mixture.create_delay_mixture(
            E=n_sources,
            S=n_mics,
            L=self.max_delay + 1,
            delay_matrix=delay_matrix,
        )


@dataclass
class AudioSceneSpec:
    """Contraintes partielles pour une scene; tout champ omis reste genere aleatoirement."""
    source_specs: list[CompositeSignalSpec] = field(default_factory=list)
    local_noise_specs: list[CompositeSignalSpec] = field(default_factory=list)
    continuous_noise_specs: list[SignalPlacementSpec] = field(default_factory=list)
    delay_matrix: np.ndarray | None = None


@dataclass
class AudioSceneMetadata:
    fs: int
    duration: float
    n_sources: int
    n_mics: int
    source_types: list[str]
    local_noise_types: list[str]
    continuous_noise_types: list[str]
    max_delay: int
    delay_matrix: np.ndarray
    seed: int | None


@dataclass
class AudioScene:
    sources: MultiSignal
    mixing: Mixture
    clean_mixed: MultiSignal
    local_noises: MultiSignal
    continuous_noises: MultiSignal
    mixed: MultiSignal
    metadata: AudioSceneMetadata


class AudioSceneGenerator:
    def __init__(
        self,
        fs: int,
        scene_duration: float,
        n_sources: int,
        n_mics: int,
        max_delay: int,
        source_placements_range: tuple[int, int] = (1, 3),
        local_noise_placements_range: tuple[int, int] = (0, 2),
        source_gain_range: tuple[float, float] = (0.5, 1.0),
        local_noise_gain_range: tuple[float, float] = (0.05, 0.3),
        continuous_noise_gain_range: tuple[float, float] = (1.0, 1.0),
        seed: int | None = None,
    ):
        self.fs = fs
        self.scene_duration = scene_duration
        self.n_sources = n_sources
        self.n_mics = n_mics
        self.max_delay = max_delay
        self.seed = seed

        self.source_generator = CompositeSignalGenerator(
            registry=_SOURCE_SIGNAL_TYPE,
            n_placements_range=source_placements_range,
            gain_range=source_gain_range,
        )
        self.local_noise_generator = CompositeSignalGenerator(
            registry=_LOCAL_NOISE_SIGNAL_TYPE,
            n_placements_range=local_noise_placements_range,
            gain_range=local_noise_gain_range,
        )
        self.continuous_noise_generator = SignalGenerator(_CONTINUOUS_NOISE_SIGNAL_TYPE)
        self.continuous_noise_gain_range = continuous_noise_gain_range
        self.mixture_generator = MixtureGenerator(max_delay=max_delay)

    def generate(self, spec: AudioSceneSpec | None = None, seed: int | None = None) -> AudioScene:
        spec = spec or AudioSceneSpec()
        rng = np.random.default_rng(self.seed if seed is None else seed)

        source_specs = self._expand_composite_specs(spec.source_specs, self.n_sources)
        sources_list = [
            self.source_generator.generate(
                rng=rng,
                freq=self.fs,
                scene_duration=self.scene_duration,
                spec=source_spec,
            )
            for source_spec in source_specs
        ]
        sources = MultiSignal(sources_list)

        mixing = self.mixture_generator.generate(
            rng=rng,
            n_sources=self.n_sources,
            n_mics=self.n_mics,
            delay_matrix=spec.delay_matrix,
        )
        clean_mixed = mixing.apply(sources, mode="same")

        local_noise_specs = self._expand_composite_specs(spec.local_noise_specs, self.n_mics)
        local_noises = MultiSignal(
            [
                self.local_noise_generator.generate(
                    rng=rng,
                    freq=self.fs,
                    scene_duration=self.scene_duration,
                    spec=noise_spec,
                )
                for noise_spec in local_noise_specs
            ]
        )

        continuous_noise_specs = self._expand_placement_specs(
            spec.continuous_noise_specs,
            self.n_mics,
        )
        continuous_noise_outputs = [
            self._generate_continuous_noise(
                rng=rng,
                spec=noise_spec,
            )
            for noise_spec in continuous_noise_specs
        ]
        continuous_noises = MultiSignal([output[0] for output in continuous_noise_outputs])

        mixed = clean_mixed + local_noises + continuous_noises
        metadata = AudioSceneMetadata(
            fs=self.fs,
            duration=self.scene_duration,
            n_sources=self.n_sources,
            n_mics=self.n_mics,
            source_types=self._composite_types(sources_list),
            local_noise_types=self._composite_types(local_noises.signals),
            continuous_noise_types=[output[1] for output in continuous_noise_outputs],
            max_delay=self.max_delay,
            delay_matrix=self._delay_matrix_from_mixture(mixing),
            seed=self.seed if seed is None else seed,
        )

        return AudioScene(
            sources=sources,
            mixing=mixing,
            clean_mixed=clean_mixed,
            local_noises=local_noises,
            continuous_noises=continuous_noises,
            mixed=mixed,
            metadata=metadata,
        )

    def _generate_continuous_noise(
        self,
        rng: np.random.Generator,
        spec: SignalPlacementSpec,
    ) -> tuple[Signal, str]:
        type_name = _random_type_name(rng, spec.signal_type, _CONTINUOUS_NOISE_SIGNAL_TYPE)
        signal_cls = _CONTINUOUS_NOISE_SIGNAL_TYPE[type_name]
        fixed_params = dict(spec.signal_params)
        fixed_params.setdefault("time_duration", self.scene_duration)
        signal = signal_cls.generate_random(
            rng=rng,
            freq=self.fs,
            scene_duration=self.scene_duration,
            fixed_params=fixed_params,
        )
        gain = _random_value(
            rng,
            spec.gain,
            self.continuous_noise_gain_range[0],
            self.continuous_noise_gain_range[1],
        )
        return signal * gain, type_name

    @staticmethod
    def _expand_composite_specs(
        specs: list[CompositeSignalSpec],
        expected_count: int,
    ) -> list[CompositeSignalSpec]:
        if len(specs) > expected_count:
            raise ValueError("Trop de CompositeSignalSpec fournis.")
        return specs + [CompositeSignalSpec() for _ in range(expected_count - len(specs))]

    @staticmethod
    def _expand_placement_specs(
        specs: list[SignalPlacementSpec],
        expected_count: int,
    ) -> list[SignalPlacementSpec]:
        if len(specs) > expected_count:
            raise ValueError("Trop de SignalPlacementSpec fournis.")
        return specs + [SignalPlacementSpec() for _ in range(expected_count - len(specs))]

    @staticmethod
    def _composite_types(signals: Sequence[Signal]) -> list[str]:
        types: list[str] = []
        for signal in signals:
            placements = getattr(signal, "placements", [])
            types.append("+".join(placement.signal.signal_type for placement in placements))
        return types

    @staticmethod
    def _delay_matrix_from_mixture(mixing: Mixture) -> np.ndarray:
        delay_matrix = np.zeros((mixing.S, mixing.E), dtype=int)
        for mic_idx in range(mixing.S):
            for source_idx in range(mixing.E):
                delay_matrix[mic_idx, source_idx] = int(
                    np.argmax(mixing.filters[mic_idx, source_idx])
                )
        return delay_matrix


register_SourceSignal(SinSignal)
register_SourceSignal(SpikeSignal)
register_LocalNoiseSignal(SpikeSignal)
register_LocalNoiseSignal(GaussianNoise)
register_ContinuousNoiseSignal(GaussianNoise)
