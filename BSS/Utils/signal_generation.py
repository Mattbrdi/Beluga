"""
Generation de signaux et de scenes acoustiques synthetiques.

Architecture du module
----------------------
1. TypedSignal represente une brique sonore mono typee. Chaque sous-classe
   (SinSignal, SpikeSignal, GaussianNoise, etc.) sait construire un signal a
   partir de parametres explicites avec generate(), ou tirer ses parametres
   manquants avec generate_random(). L'amplitude des briques est normalisee par
   defaut; leur niveau dans une scene est principalement controle par un gain.

2. SignalPlacement place un TypedSignal sur une timeline. Il porte le temps de
   debut, la fenetre et le gain, puis render() produit le signal decale et
   fenetre. SignalPlacementSpec permet de fixer seulement certains de ces
   champs; les champs non fixes sont tires aleatoirement par le generateur.

3. CompositeSignal represente une piste mono de frequence et de duree fixes.
   Il contient plusieurs SignalPlacement compatibles en frequence. render()
   additionne leurs contributions et retourne toujours exactement la duree du
   composite, en completant avec des zeros ou en tronquant ce qui depasse.

4. Les registres associent un nom de type a une classe TypedSignal. Des
   registres distincts existent pour les sources, les bruits locaux et les
   bruits continus. Ils peuvent etre etendus avec les fonctions register_* ou
   remplaces lors de la construction d'un AudioSceneGenerator.

5. SignalPlacementGenerator et CompositeSignalGenerator appliquent les specs.
   Une spec peut fixer un type, limiter le tirage a une liste de types, fixer
   certains parametres du signal ou laisser chaque choix au hasard. Toutes les
   operations aleatoires utilisent un np.random.Generator pour permettre la
   reproductibilite par graine.

6. AudioSceneGenerator orchestre la generation complete:
   - creation de n sources sous forme de CompositeSignal;
   - rendu des sources dans un MultiSignal;
   - generation d'une Mixture de retards source-vers-micro;
   - calcul du melange propre sur m microphones;
   - generation des bruits locaux et continus de chaque microphone;
   - somme du melange propre et du bruit pour produire le signal final.

7. AudioScene contient les donnees utiles a l'exploitation: sources propres,
   matrice de melange, melange propre, bruit total et melange final. Les
   dataclasses de metadata conservent les parametres de generation necessaires
   pour analyser ou reproduire une scene sans dupliquer toutes les donnees
   intermediaires.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
from dataclasses import dataclass, field
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal
from collections.abc import Sequence

from .signal_class import Signal, MultiSignal, Mixture


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


class TypedSignal(Signal, ABC):
    signal_type: str = "generic"
    allowed_windows: tuple[WindowName, ...] = (None, "hann")
    default_window: WindowName = "hann"

    def __init__(self, data: np.ndarray, freq: float):
        super().__init__(data, freq)

    @classmethod
    def validate_fixed_params(cls, fixed_params: dict[str, Any] | None) -> None:
        """Verifie que les parametres fixes sont acceptes par generate()."""
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
    def generate(cls, **params: Any) -> "TypedSignal":
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
        params = cls.generate_random_params(
            rng=rng,
            freq=freq,
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
                1.0,
            ),
        }

    @classmethod
    def generate_random(
        cls,
        rng: np.random.Generator,
        freq: float,
        fixed_params: dict[str, Any] | None = None,
    ) -> "GaussianNoise":
        params = cls.generate_random_params(rng, freq, fixed_params)
        return cls.generate_with_rng(rng=rng, **params)


@dataclass
class SignalPlacementSpec:
    """Contraintes partielles pour un placement: les champs None/random sont tires aleatoirement."""
    signal_type: SignalTypeChoice = None
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
        if self.start_time < 0:
            raise ValueError("start_time doit etre positif ou nul.")
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
        latency = Signal(
            data=np.zeros(int(round(self.start_time * self.signal.freq))),
            freq=self.signal.freq,
        )
        return Signal.concat(latency, data_array)

    def _apply_window(self, signal: Signal) -> Signal:
        if self.window is None:
            return signal.copy()
        window = Signal(
            data=sp_signal.get_window(window=self.window, Nx=len(signal.data)),
            freq=signal.freq,
        )
        return signal * window


class CompositeSignal:
    """Piste de duree fixe composee de plusieurs SignalPlacement."""

    def __init__(
        self,
        placements: list[SignalPlacement],
        freq: float,
        duration: float,
    ):
        if freq <= 0:
            raise ValueError("freq doit etre strictement positive.")
        if duration < 0:
            raise ValueError("duration doit etre positive ou nulle.")

        self.freq = float(freq)
        self.duration = float(duration)
        self.placements: list[SignalPlacement] = []
        for placement in placements:
            self.add_placement(placement)

    @staticmethod
    def verify_frequency_coherence(placements: list[SignalPlacement]) -> bool:
        if len(placements) <= 1:
            return True
        reference = placements[0]
        return all(reference.is_compatible_with(place) for place in placements[1:])

    def render(self) -> Signal:
        n_samples = int(round(self.duration * self.freq))
        data = np.zeros(n_samples)

        for placement in self.placements:
            rendered = placement.render()
            usable_length = min(len(rendered.data), n_samples)
            data[:usable_length] += rendered.data[:usable_length]

        return Signal(data=data, freq=self.freq)

    def describe_and_plot(
        self,
        title: str = "CompositeSignal",
        figsize: tuple[float, float] = (12, 3),
        **plot_kwargs: Any,
    ) -> Signal:
        """Affiche les placements, trace le composite et retourne son rendu."""
        for index, placement in enumerate(self.placements):
            signal = placement.signal
            print(
                f"placement {index}: type={signal.signal_type}, "
                f"start={placement.start_time:.3f} s, "
                f"duration={signal.duration:.3f} s, "
                f"window={placement.window}, gain={placement.gain:.3f}"
            )

        rendered = self.render()
        plot_kwargs.setdefault("linewidth", 0.8)
        fig, _ = rendered.plot(title=title, **plot_kwargs)
        fig.set_size_inches(figsize)
        plt.show()
        return rendered

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
        target_len = end_idx - start_idx

        data = signal.data[start_idx:end_idx].copy()

        if len(data) < target_len:
            data = np.pad(data, (0, target_len - len(data)))

        return Signal(data, signal.freq)
    def add_placement(self, placement: SignalPlacement) -> None:
        if placement.signal.freq != self.freq:
            raise ValueError("SignalPlacement non compatible avec la frequence du composite.")
        self.placements.append(placement)


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


class SignalPlacementGenerator:
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
        signal_cls.validate_fixed_params(spec.signal_params)
        signal = signal_cls.generate_random(
            rng=rng,
            freq=freq,
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
        self.signal_generator = SignalPlacementGenerator(registry)
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
            warnings.warn(
                "Plus de SignalPlacementSpec fournis que n_placements; "
                "les specs explicites sont prioritaires.",
                UserWarning,
                stacklevel=2,
            )
            n_placements = len(placement_specs)
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
        return CompositeSignal(
            placements=placements,
            freq=freq,
            duration=scene_duration,
        )


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
class SignalPlacementMetadata:
    signal_type: str
    signal_params: dict[str, Any]
    start_time: float
    duration: float
    window: WindowName
    gain: float


@dataclass
class CompositeSignalMetadata:
    placements: list[SignalPlacementMetadata]
    duration: float


@dataclass
class ContinuousNoiseMetadata:
    signal_type: str
    signal_params: dict[str, Any]
    gain: float


@dataclass
class AudioSceneMetadata:
    fs: int
    duration: float
    n_sources: int
    n_mics: int
    source_types: list[str]
    local_noise_types: list[str]
    continuous_noise_types: list[str]
    source_composites: list[CompositeSignalMetadata]
    local_noise_composites: list[CompositeSignalMetadata]
    continuous_noises: list[ContinuousNoiseMetadata]
    max_delay: int
    delay_matrix: np.ndarray
    seed: int | None


@dataclass
class AudioScene:
    sources: MultiSignal
    mixing: Mixture
    clean_mixed: MultiSignal
    noise: MultiSignal
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
        source_placements_range: tuple[int, int] = (1, 3),  # Nombre min/max de SignalPlacement par source.
        local_noise_placements_range: tuple[int, int] = (0, 2),  # Nombre min/max de SignalPlacement par bruit local micro.
        source_gain_range: tuple[float, float] = (0.5, 1.0),
        local_noise_gain_range: tuple[float, float] = (0.05, 0.3),
        continuous_noise_gain_range: tuple[float, float] = (1.0, 1.0),
        source_registry: dict[str, type[TypedSignal]] | None = None,
        local_noise_registry: dict[str, type[TypedSignal]] | None = None,
        continuous_noise_registry: dict[str, type[TypedSignal]] | None = None,
        seed: int | None = None,
    ):
        self.fs = fs
        self.scene_duration = scene_duration
        self.n_sources = n_sources
        self.n_mics = n_mics
        self.max_delay = max_delay
        self.seed = seed
        self.source_registry = dict(_SOURCE_SIGNAL_TYPE if source_registry is None else source_registry)
        self.local_noise_registry = dict(
            _LOCAL_NOISE_SIGNAL_TYPE if local_noise_registry is None else local_noise_registry
        )
        self.continuous_noise_registry = dict(
            _CONTINUOUS_NOISE_SIGNAL_TYPE
            if continuous_noise_registry is None
            else continuous_noise_registry
        )

        self.source_generator = CompositeSignalGenerator(
            registry=self.source_registry,
            n_placements_range=source_placements_range,
            gain_range=source_gain_range,
        )
        self.local_noise_generator = CompositeSignalGenerator(
            registry=self.local_noise_registry,
            n_placements_range=local_noise_placements_range,
            gain_range=local_noise_gain_range,
        )
        self.continuous_noise_generator = SignalPlacementGenerator(self.continuous_noise_registry) #Un seul type de bruit continue
        self.continuous_noise_gain_range = continuous_noise_gain_range
        self.mixture_generator = MixtureGenerator(max_delay=max_delay)

    def generate(self, spec: AudioSceneSpec | None = None, seed: int | None = None) -> AudioScene:
        spec = spec or AudioSceneSpec()
        rng = np.random.default_rng(self.seed if seed is None else seed)

        source_specs = self._expand_composite_specs(spec.source_specs, self.n_sources)
        source_composites = [
            self.source_generator.generate(
                rng=rng,
                freq=self.fs,
                scene_duration=self.scene_duration,
                spec=source_spec,
            )
            for source_spec in source_specs
        ]
        sources = MultiSignal([source.render() for source in source_composites])

        mixing = self.mixture_generator.generate(
            rng=rng,
            n_sources=self.n_sources,
            n_mics=self.n_mics,
            delay_matrix=spec.delay_matrix,
        )
        clean_mixed = mixing.apply(sources, mode="same")

        local_noise_specs = self._expand_composite_specs(spec.local_noise_specs, self.n_mics)
        local_noise_composites = [
            self.local_noise_generator.generate(
                rng=rng,
                freq=self.fs,
                scene_duration=self.scene_duration,
                spec=noise_spec,
            )
            for noise_spec in local_noise_specs
        ]
        local_noises = MultiSignal([noise.render() for noise in local_noise_composites])

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

        noise = local_noises + continuous_noises
        mixed = clean_mixed + noise
        metadata = AudioSceneMetadata(
            fs=self.fs,
            duration=self.scene_duration,
            n_sources=self.n_sources,
            n_mics=self.n_mics,
            source_types=self._composite_types(source_composites),
            local_noise_types=self._composite_types(local_noise_composites),
            continuous_noise_types=[output[1].signal_type for output in continuous_noise_outputs],
            source_composites=[
                self._composite_metadata(source)
                for source in source_composites
            ],
            local_noise_composites=[
                self._composite_metadata(noise)
                for noise in local_noise_composites
            ],
            continuous_noises=[output[1] for output in continuous_noise_outputs],
            max_delay=self.max_delay,
            delay_matrix=self._delay_matrix_from_mixture(mixing),
            seed=self.seed if seed is None else seed,
        )

        return AudioScene(
            sources=sources,
            mixing=mixing,
            clean_mixed=clean_mixed,
            noise=noise,
            mixed=mixed,
            metadata=metadata,
        )

    def _generate_continuous_noise(
        self,
        rng: np.random.Generator,
        spec: SignalPlacementSpec,
    ) -> tuple[Signal, ContinuousNoiseMetadata]:
        type_name = _random_type_name(rng, spec.signal_type, self.continuous_noise_registry)
        signal_cls = self.continuous_noise_registry[type_name]
        fixed_params = dict(spec.signal_params)
        signal_cls.validate_fixed_params(fixed_params)
        fixed_params.setdefault("time_duration", self.scene_duration)
        signal = signal_cls.generate_random(
            rng=rng,
            freq=self.fs,
            fixed_params=fixed_params,
        )
        gain = _random_value(
            rng,
            spec.gain,
            self.continuous_noise_gain_range[0],
            self.continuous_noise_gain_range[1],
        )
        metadata = ContinuousNoiseMetadata(
            signal_type=type_name,
            signal_params=self._signal_params(signal),
            gain=gain,
        )
        return signal * gain, metadata

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
    def _composite_types(composites: Sequence[CompositeSignal]) -> list[str]:
        types: list[str] = []
        for composite in composites:
            types.append("+".join(placement.signal.signal_type for placement in composite.placements))
        return types

    @classmethod
    def _composite_metadata(cls, composite: CompositeSignal) -> CompositeSignalMetadata:
        return CompositeSignalMetadata(
            placements=[
                cls._placement_metadata(placement)
                for placement in composite.placements
            ],
            duration=composite.duration,
        )

    @classmethod
    def _placement_metadata(cls, placement: SignalPlacement) -> SignalPlacementMetadata:
        return SignalPlacementMetadata(
            signal_type=placement.signal.signal_type,
            signal_params=cls._signal_params(placement.signal),
            start_time=placement.start_time,
            duration=placement.signal.duration,
            window=placement.window,
            gain=placement.gain,
        )

    @staticmethod
    def _signal_params(signal: TypedSignal) -> dict[str, Any]:
        ignored_fields = {"data", "freq"}
        params: dict[str, Any] = {}
        for key, value in vars(signal).items():
            if key in ignored_fields:
                continue
            if isinstance(value, np.generic):
                value = value.item()
            elif isinstance(value, np.ndarray):
                value = value.tolist()
            params[key] = value
        return params

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
