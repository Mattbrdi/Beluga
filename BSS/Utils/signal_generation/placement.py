"""Placement temporel et composition de signaux mono."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal

from ..signal_class import Signal
from .common import RANDOM_WINDOW, SignalTypeChoice, WindowChoice, WindowName
from .TypedSignal import TypedSignal

@dataclass
class SignalPlacementSpec:
    """Contraintes partielles pour un placement.

    signal_type=None herite des types autorises par le composite ou la scene.
    window="random" tire une fenetre autorisee; window=None desactive le
    fenetrage. Les autres champs None sont tires aleatoirement.
    """
    signal_type: SignalTypeChoice = None
    signal_params: dict[str, Any] = field(default_factory=dict)
    start_time: float | None = None
    window: WindowChoice = RANDOM_WINDOW
    gain: float | None = None


@dataclass
class CompositeSignalSpec:
    """Contraintes partielles pour un composite et ses placements.

    n_placements fixe le nombre cible lorsqu'il est renseigne. Les specs de
    placement explicites restent prioritaires: si elles sont plus nombreuses,
    elles constituent le nombre final de placements.
    allowed_signal_types sert de valeur par defaut aux placements qui ne fixent
    pas eux-memes leur type.
    """
    n_placements: int | None = None
    allowed_signal_types: SignalTypeChoice = None
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

