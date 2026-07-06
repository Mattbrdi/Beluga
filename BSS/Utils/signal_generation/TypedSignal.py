"""Compatibilité avec l'ancien module; préférer les imports du package public."""

from .signals import GaussianNoise, SinSignal, SpikeSignal, TypedSignal, WhistleSignal

__all__ = [
    "GaussianNoise",
    "SinSignal",
    "SpikeSignal",
    "TypedSignal",
    "WhistleSignal",
]
