"""Generation aleatoire des placements et signaux composites."""
from __future__ import annotations

from dataclasses import replace
import warnings

import numpy as np

from .common import SignalTypeChoice, RANDOM_WINDOW, _random_choice, _random_type_name, _random_value
from .placement import CompositeSignal, CompositeSignalSpec, SignalPlacement, SignalPlacementSpec
from .TypedSignal import TypedSignal

class SignalPlacementGenerator:
    def __init__(self, registry: dict[str, type[TypedSignal]]):
        for signal_cls in registry.values():
            if not issubclass(signal_cls, TypedSignal):
                raise TypeError("Toutes les classes du registry doivent heriter de TypedSignal.")
            signal_cls.validate_generation_contract()
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
        placement_rate: float,
        gain_range: tuple[float, float],
        min_placements: int = 0,
    ):
        if placement_rate < 0:
            raise ValueError("placement_rate doit etre positif ou nul.")
        if min_placements < 0:
            raise ValueError("min_placements doit etre positif ou nul.")

        self.signal_placement_generator = SignalPlacementGenerator(registry)
        self.placement_rate = float(placement_rate)
        self.gain_range = gain_range
        self.min_placements = int(min_placements)

    def generate(
        self,
        rng: np.random.Generator,
        freq: float,
        scene_duration: float,
        spec: CompositeSignalSpec | None = None,
        allowed_signal_types: SignalTypeChoice = None,
    ) -> CompositeSignal:
        if scene_duration < 0:
            raise ValueError("scene_duration doit etre positive ou nulle.")
        spec = spec or CompositeSignalSpec()
        if spec.n_placements is not None:
            if spec.n_placements < 0:
                raise ValueError("n_placements doit etre positif ou nul.")
            n_placements = int(spec.n_placements)
        else:
            n_placements = max(
                self.min_placements,
                int(rng.poisson(self.placement_rate * scene_duration)),
            )

        default_signal_types = self._select_default_signal_types(
            composite_signal_types=spec.allowed_signal_types,
            scene_signal_types=allowed_signal_types,
        )
        placement_specs = self._prepare_placement_specs(
            spec=spec,
            n_placements=n_placements,
            default_signal_types=default_signal_types,
        )

        placements = [
            self.signal_placement_generator.generate_placement(
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

    @staticmethod
    def _select_default_signal_types(
        composite_signal_types: SignalTypeChoice,
        scene_signal_types: SignalTypeChoice,
    ) -> SignalTypeChoice:
        """Donne la priorite aux types autorises par le composite."""
        if composite_signal_types is not None:
            return composite_signal_types
        return scene_signal_types

    @staticmethod
    def _apply_default_signal_types(
        placement_spec: SignalPlacementSpec,
        default_signal_types: SignalTypeChoice,
    ) -> SignalPlacementSpec:
        """Fait heriter les types par defaut sans modifier la spec originale."""
        if placement_spec.signal_type is not None:
            return placement_spec
        return replace(placement_spec, signal_type=default_signal_types)

    @classmethod
    def _prepare_placement_specs(
        cls,
        spec: CompositeSignalSpec,
        n_placements: int,
        default_signal_types: SignalTypeChoice,
    ) -> list[SignalPlacementSpec]:
        """Resout les specs explicites puis complete les placements manquants."""
        placement_specs = [
            cls._apply_default_signal_types(placement_spec, default_signal_types)
            for placement_spec in spec.placements
        ]

        if len(placement_specs) > n_placements and spec.n_placements is not None:
            warnings.warn(
                "Plus de SignalPlacementSpec fournis que n_placements; "
                "les specs explicites sont prioritaires.",
                UserWarning,
                stacklevel=3,
            )

        target_count = max(n_placements, len(placement_specs))
        missing_count = target_count - len(placement_specs)
        placement_specs.extend(
            SignalPlacementSpec(signal_type=default_signal_types)
            for _ in range(missing_count)
        )
        return placement_specs
