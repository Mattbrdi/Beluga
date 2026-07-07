"""Compatibilité avec l'ancien module; préférer les imports du package public."""

from .signals import (
    GaussianNoise,
    LargeShipNoise,
    SinSignal,
    SpikeSignal,
    TypedSignal,
    WhistleSignal,
)

__all__ = [
    "GaussianNoise",
    "LargeShipNoise",
    "SinSignal",
    "SpikeSignal",
    "TypedSignal",
    "WhistleSignal",
]
