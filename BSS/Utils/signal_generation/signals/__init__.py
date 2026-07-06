"""Signaux typés disponibles pour construire des scènes synthétiques."""

from .base import TypedSignal
from .gaussian_noise import GaussianNoise
from .sine import SinSignal
from .spike import SpikeSignal
from .whistle import WhistleSignal

__all__ = [
    "GaussianNoise",
    "SinSignal",
    "SpikeSignal",
    "TypedSignal",
    "WhistleSignal",
]
