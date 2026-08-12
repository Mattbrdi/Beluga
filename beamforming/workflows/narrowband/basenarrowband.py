import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert

from beamforming.config import *

from beamforming.grid import SearchGrid
from beamforming.geometry import TetrahedralArray
from beamforming.results import PseudoSpectrumResult
from beamforming.beamformers.base import Beamformer
from beamforming.workflows.workflow import BeamformingWorkflow

from beamforming.steering import compute_steering_vector
from beamforming.signal.covariance import spatial_covariance


class NarrowbandBeamformer(BeamformingWorkflow):
    def __init__(self, tetrahedra: TetrahedralArray, beamformer: Beamformer):
        super().__init__(tetrahedra, beamformer)

    def compute(
        self, signal: NDArray[np.float64], grid: SearchGrid, fc: float
    ) -> PseudoSpectrumResult:
        steering_vector = compute_steering_vector(self.tetrahedra, grid, fc)
        
        analytic_signal = hilbert(signal, axis=-1)
        correlation_matrix = spatial_covariance(analytic_signal)[None, ...]

        pseudo_spectrum = self.beamformer.compute_pseudo_spectrum(steering_vector, correlation_matrix)

        return PseudoSpectrumResult(grid, pseudo_spectrum[0])
