from abc import ABC, abstractmethod


import numpy as np
from numpy.typing import NDArray

from beamforming.geometry import TetrahedralArray, Direction
from beamforming.grid import SearchGrid
from beamforming.results import PseudoSpectrumResult
from beamforming.beamformers.base import Beamformer


class BeamformingWorkflow(ABC):
    def __init__(self, tetrahedra: TetrahedralArray, beamformer: Beamformer):
        super().__init__()
        self.tetrahedra = tetrahedra
        self.beamformer = beamformer

    @abstractmethod
    def compute(
        self, signal: NDArray[np.float64], grid: SearchGrid
    ) -> PseudoSpectrumResult: ...

    def estimate_doa(
        self,
        signal: NDArray[np.float64],
        grid: SearchGrid,
    ) -> Direction:
        return self.compute(signal, grid).doa
