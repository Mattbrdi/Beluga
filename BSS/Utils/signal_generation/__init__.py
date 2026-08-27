"""Generation de signaux et de scenes acoustiques synthetiques.

Ce module reexporte l'API publique historique afin que les imports depuis
``BSS.Utils.signal_generation`` restent compatibles apres le decoupage.
"""

from .common import RANDOM_WINDOW, SignalTypeChoice, WindowChoice, WindowName
from .generators import CompositeSignalGenerator, SignalPlacementGenerator
from .mixture import MixtureGenerator
from .placement import (
    CompositeSignal,
    CompositeSignalSpec,
    SignalPlacement,
    SignalPlacementSpec,
)
from .registries import (
    register_ContinuousNoiseSignal,
    register_LocalNoiseSignal,
    register_SourceSignal,
)
from .scene import (
    AudioScene,
    AudioSceneGenerator,
    AudioSceneMetadata,
    AudioSceneSpec,
    CompositeSignalMetadata,
    ContinuousNoiseMetadata,
    SignalPlacementMetadata,
)
from .signals import (
    GaussianNoise,
    LargeShipNoise,
    SinSignal,
    SpikeSignal,
    TypedSignal,
    WhistleSignal,
)


register_SourceSignal(SinSignal)
register_SourceSignal(SpikeSignal)
register_SourceSignal(WhistleSignal)
register_SourceSignal(LargeShipNoise)
register_LocalNoiseSignal(SpikeSignal)
register_ContinuousNoiseSignal(GaussianNoise)


__all__ = [
    "AudioScene",
    "AudioSceneGenerator",
    "AudioSceneMetadata",
    "AudioSceneSpec",
    "CompositeSignal",
    "CompositeSignalGenerator",
    "CompositeSignalMetadata",
    "CompositeSignalSpec",
    "ContinuousNoiseMetadata",
    "GaussianNoise",
    "LargeShipNoise",
    "MixtureGenerator",
    "RANDOM_WINDOW",
    "SignalPlacement",
    "SignalPlacementGenerator",
    "SignalPlacementMetadata",
    "SignalPlacementSpec",
    "SignalTypeChoice",
    "SinSignal",
    "SpikeSignal",
    "TypedSignal",
    "WhistleSignal",
    "WindowChoice",
    "WindowName",
    "register_ContinuousNoiseSignal",
    "register_LocalNoiseSignal",
    "register_SourceSignal",
]
