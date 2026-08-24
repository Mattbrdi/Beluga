"""Compatibility wrapper for the Sawada package.

New code should import from ``BSS.Algo_Separation.Sawada``.
"""

from .Sawada import (
    EMClustering,
    SawadaBSS,
    SawadaDebugArtifacts,
    SawadaResult,
)

__all__ = [
    "EMClustering",
    "SawadaBSS",
    "SawadaDebugArtifacts",
    "SawadaResult",
]
