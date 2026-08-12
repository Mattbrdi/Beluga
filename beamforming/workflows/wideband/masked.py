import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert

from time_frequency_mask.configuration import MAX_TDOA_IDX
from time_frequency_mask.tdoa_estimation.blob import Blob


from beamforming.config import *

from beamforming.geometry import TetrahedralArray
from beamforming.grid import SearchGrid
from beamforming.results import PseudoSpectrumResult
from beamforming.beamformers.base import Beamformer
from beamforming.workflows.workflow import BeamformingWorkflow

from beamforming.signal.covariance import spatial_covariance
from beamforming.signal.filtering import bandpass_filter
from beamforming.steering import compute_steering_vector


class MaskedBeamformer(BeamformingWorkflow):
    def __init__(self, tetrahedra: TetrahedralArray, beamformer: Beamformer):
        self.tetrahedra = tetrahedra
        self.beamformer = beamformer

    def compute(self, signal: NDArray[np.float64], grid: SearchGrid, blobs: list[Blob]):
        N = signal.shape[-1]

        Powers = []

        for blob in blobs:
            tmin_idx = int(max(0, blob.tmin_idx - MAX_TDOA_IDX))
            tmax_idx = int(min(N, blob.tmax_idx + MAX_TDOA_IDX))

            fmin = blob.fmin
            fmax = blob.fmax

            fc = (fmin + fmax) / 2

            segment = signal[:, tmin_idx:tmax_idx]
            segment = bandpass_filter(segment, SAMPLING_RATE, fmin, fmax)

            steering_vector = compute_steering_vector(self.tetrahedra, grid, fc)

            analytic_segment = hilbert(segment, axis=-1)

            correlation_matrix = spatial_covariance(analytic_segment)
            power_dB = self.beamformer.compute_pseudo_spectrum(
                steering_vector, correlation_matrix[None, ...]
            )
            Powers.append(power_dB[0])
        return PseudoSpectrumResult(grid, np.stack(Powers, axis=0).mean(axis=0))
