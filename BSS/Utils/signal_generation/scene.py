"""Modeles et orchestration d une scene audio complete."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..signal_class import Mixture, MultiSignal, Signal
from .common import SignalTypeChoice, _random_type_name, _random_value, WindowName 
from .generators import CompositeSignalGenerator, SignalPlacementGenerator
from .mixture import MixtureGenerator
from .placement import CompositeSignal, CompositeSignalSpec, SignalPlacement, SignalPlacementSpec
from .registries import _CONTINUOUS_NOISE_SIGNAL_TYPE, _LOCAL_NOISE_SIGNAL_TYPE, _SOURCE_SIGNAL_TYPE
from .TypedSignal import TypedSignal

@dataclass
class AudioSceneSpec:
    """Contraintes partielles appliquees a une generation de scene.

    Les types sont resolus par priorite: placement, composite, scene, registre.
    snr_db fixe le rapport en dB entre l'energie totale du melange propre sur
    les micros et celle des bruits continus; les bruits locaux sont exclus.
    """
    allowed_source_signal_types: SignalTypeChoice = None
    allowed_local_noise_signal_types: SignalTypeChoice = None
    allowed_continuous_noise_signal_types: SignalTypeChoice = None
    snr_db: float | None = None
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
    """Description des placements et energie du composite rendu."""
    placements: list[SignalPlacementMetadata]
    duration: float
    energy: float | None = None


@dataclass
class ContinuousNoiseMetadata:
    """Description d'un bruit continu apres application de son gain et du SNR."""
    signal_type: str
    signal_params: dict[str, Any]
    gain: float
    energy: float | None = None


@dataclass
class AudioSceneMetadata:
    """Parametres et valeurs realisees necessaires pour analyser une scene."""
    fs: int
    duration: float
    n_sources: int
    n_mics: int
    source_composites: list[CompositeSignalMetadata]
    local_noise_composites: list[CompositeSignalMetadata]
    continuous_noises: list[ContinuousNoiseMetadata]
    max_delay: int
    delay_matrix: np.ndarray
    seed: int | None
    snr_db: float | None = None


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
        source_placement_rate: float = 0.75,
        local_noise_placement_rate: float = 1.0 / 3.0,
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
            placement_rate=source_placement_rate,
            gain_range=source_gain_range,
            min_placements=1,
        )
        self.local_noise_generator = CompositeSignalGenerator(
            registry=self.local_noise_registry,
            placement_rate=local_noise_placement_rate,
            gain_range=local_noise_gain_range,
            min_placements=0,
        )
        self.continuous_noise_generator = SignalPlacementGenerator(
            self.continuous_noise_registry
        )
        self.continuous_noise_gain_range = continuous_noise_gain_range
        self.mixture_generator = MixtureGenerator(max_delay=max_delay)

    def generate(self, spec: AudioSceneSpec | None = None, seed: int | None = None) -> AudioScene:
        spec = spec or AudioSceneSpec()
        effective_seed = self.seed if seed is None else seed
        rng = np.random.default_rng(effective_seed)

        source_composites, sources = self._generate_sources(rng, spec)
        mixing, clean_mixed = self._mix_sources(rng, sources, spec)
        local_noise_composites, local_noises = self._generate_local_noises(rng, spec)
        continuous_noises, continuous_noise_metadata = self._generate_continuous_noises(
            rng,
            spec,
            clean_mixed,
        )

        noise = local_noises + continuous_noises
        mixed = clean_mixed + noise
        metadata = self._build_scene_metadata(
            spec=spec,
            seed=effective_seed,
            mixing=mixing,
            source_composites=source_composites,
            local_noise_composites=local_noise_composites,
            continuous_noise_metadata=continuous_noise_metadata,
        )

        return AudioScene(
            sources=sources,
            mixing=mixing,
            clean_mixed=clean_mixed,
            noise=noise,
            mixed=mixed,
            metadata=metadata,
        )

    def _generate_sources(
        self,
        rng: np.random.Generator,
        spec: AudioSceneSpec,
    ) -> tuple[list[CompositeSignal], MultiSignal]:
        """Genere et rend les sources propres de la scene."""
        source_specs = self._expand_composite_specs(spec.source_specs, self.n_sources)
        source_composites = [
            self.source_generator.generate(
                rng=rng,
                freq=self.fs,
                scene_duration=self.scene_duration,
                spec=source_spec,
                allowed_signal_types=spec.allowed_source_signal_types,
            )
            for source_spec in source_specs
        ]
        sources = MultiSignal([source.render() for source in source_composites])
        return source_composites, sources

    def _mix_sources(
        self,
        rng: np.random.Generator,
        sources: MultiSignal,
        spec: AudioSceneSpec,
    ) -> tuple[Mixture, MultiSignal]:
        """Genere la mixture et propage les sources sur les microphones."""
        mixing = self.mixture_generator.generate(
            rng=rng,
            n_sources=self.n_sources,
            n_mics=self.n_mics,
            delay_matrix=spec.delay_matrix,
        )
        clean_mixed = mixing.apply(sources, mode="same")
        return mixing, clean_mixed

    def _generate_local_noises(
        self,
        rng: np.random.Generator,
        spec: AudioSceneSpec,
    ) -> tuple[list[CompositeSignal], MultiSignal]:
        """Genere et rend le bruit local propre a chaque microphone."""
        local_noise_specs = self._expand_composite_specs(spec.local_noise_specs, self.n_mics)
        local_noise_composites = [
            self.local_noise_generator.generate(
                rng=rng,
                freq=self.fs,
                scene_duration=self.scene_duration,
                spec=noise_spec,
                allowed_signal_types=spec.allowed_local_noise_signal_types,
            )
            for noise_spec in local_noise_specs
        ]
        local_noises = MultiSignal([noise.render() for noise in local_noise_composites])
        return local_noise_composites, local_noises

    def _generate_continuous_noises(
        self,
        rng: np.random.Generator,
        spec: AudioSceneSpec,
        clean_mixed: MultiSignal,
    ) -> tuple[MultiSignal, list[ContinuousNoiseMetadata]]:
        """Genere les bruits continus et applique le SNR cible si demande."""
        continuous_noise_specs = self._expand_placement_specs(
            spec.continuous_noise_specs,
            self.n_mics,
        )
        continuous_noise_outputs = [
            self._generate_continuous_noise(
                rng=rng,
                spec=noise_spec,
                allowed_signal_types=spec.allowed_continuous_noise_signal_types,
            )
            for noise_spec in continuous_noise_specs
        ]
        if spec.snr_db is not None:
            continuous_noise_outputs = self._apply_continuous_noise_snr(
                clean_mixed=clean_mixed,
                continuous_noise_outputs=continuous_noise_outputs,
                snr_db=spec.snr_db,
            )
        continuous_noises = MultiSignal([output[0] for output in continuous_noise_outputs])
        metadata = [output[1] for output in continuous_noise_outputs]
        return continuous_noises, metadata

    def _build_scene_metadata(
        self,
        spec: AudioSceneSpec,
        seed: int | None,
        mixing: Mixture,
        source_composites: list[CompositeSignal],
        local_noise_composites: list[CompositeSignal],
        continuous_noise_metadata: list[ContinuousNoiseMetadata],
    ) -> AudioSceneMetadata:
        """Construit les metadonnees finales a partir des objets generes."""
        return AudioSceneMetadata(
            fs=self.fs,
            duration=self.scene_duration,
            n_sources=self.n_sources,
            n_mics=self.n_mics,
            source_composites=[
                self._composite_metadata(source)
                for source in source_composites
            ],
            local_noise_composites=[
                self._composite_metadata(noise)
                for noise in local_noise_composites
            ],
            continuous_noises=continuous_noise_metadata,
            max_delay=self.max_delay,
            delay_matrix=self._delay_matrix_from_mixture(mixing),
            seed=seed,
            snr_db=None if spec.snr_db is None else float(spec.snr_db),
        )

    def _generate_continuous_noise(
        self,
        rng: np.random.Generator,
        spec: SignalPlacementSpec,
        allowed_signal_types: SignalTypeChoice = None,
    ) -> tuple[Signal, ContinuousNoiseMetadata]:
        signal_types = (
            spec.signal_type
            if spec.signal_type is not None
            else allowed_signal_types
        )
        type_name = _random_type_name(rng, signal_types, self.continuous_noise_registry)
        signal_cls = self.continuous_noise_registry[type_name]
        fixed_params = dict(spec.signal_params)
        # Un bruit continu couvre toujours la scene complete.
        fixed_params["time_duration"] = self.scene_duration
        signal_cls.validate_fixed_params(fixed_params)
        signal = signal_cls.generate_random(
            rng=rng,
            freq=self.fs,
            **fixed_params,
        )
        gain = _random_value(
            rng,
            spec.gain,
            self.continuous_noise_gain_range[0],
            self.continuous_noise_gain_range[1],
        )
        noise = signal * gain
        metadata = ContinuousNoiseMetadata(
            signal_type=type_name,
            signal_params=self._signal_params(signal),
            gain=gain,
            energy=float(noise.energy),
        )
        return noise, metadata

    @staticmethod
    def _apply_continuous_noise_snr(
        clean_mixed: MultiSignal,
        continuous_noise_outputs: list[tuple[Signal, ContinuousNoiseMetadata]],
        snr_db: float,
    ) -> list[tuple[Signal, ContinuousNoiseMetadata]]:
        """Ajuste les bruits continus avec un gain commun pour atteindre le SNR global."""
        if not np.isfinite(snr_db):
            raise ValueError("snr_db doit etre une valeur finie.")

        signal_energy = sum(float(signal.energy) for signal in clean_mixed.signals)
        noise_energy = sum(float(signal.energy) for signal, _ in continuous_noise_outputs)
        if signal_energy <= 0:
            raise ValueError("Impossible de fixer le SNR: l'energie du signal est nulle.")
        if noise_energy <= 0:
            raise ValueError("Impossible de fixer le SNR: l'energie du bruit continu est nulle.")

        target_ratio = 10.0 ** (float(snr_db) / 10.0)
        noise_scale = np.sqrt(signal_energy / (target_ratio * noise_energy))

        scaled_outputs: list[tuple[Signal, ContinuousNoiseMetadata]] = []
        for noise, metadata in continuous_noise_outputs:
            scaled_noise = noise * noise_scale
            scaled_metadata = replace(
                metadata,
                gain=float(metadata.gain * noise_scale),
                energy=float(scaled_noise.energy),
            )
            scaled_outputs.append((scaled_noise, scaled_metadata))
        return scaled_outputs

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

    @classmethod
    def _composite_metadata(cls, composite: CompositeSignal) -> CompositeSignalMetadata:
        rendered = composite.render()
        return CompositeSignalMetadata(
            placements=[
                cls._placement_metadata(placement)
                for placement in composite.placements
            ],
            duration=composite.duration,
            energy=float(rendered.energy),
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
