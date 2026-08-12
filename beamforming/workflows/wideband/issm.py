import numpy as np
from numpy.typing import NDArray


from beamforming.config import *

from beamforming.geometry import TetrahedralArray
from beamforming.grid import SearchGrid
from beamforming.results import PseudoSpectrumResult
from beamforming.signal.stft import STFTConfig
from beamforming.beamformers.base import Beamformer
from beamforming.workflows.wideband.basewideband import WidebandBeamformer

from beamforming.signal.covariance import spatial_covariance
from beamforming.steering import compute_steering_vector


class ISSM(WidebandBeamformer):
    def __init__(
        self,
        tetrahedra: TetrahedralArray,
        beamformer: Beamformer,
        stft_config: STFTConfig,
    ):
        super().__init__(tetrahedra, beamformer, stft_config)
        self.beamformer = beamformer

    def compute_from_stft(
        self, stft: NDArray[np.complex128], freqs: NDArray[np.float64], grid: SearchGrid
    ) -> PseudoSpectrumResult:
        steering_vector = compute_steering_vector(
            self.tetrahedra, grid, freqs
        )  # (F, M, T, P)

        covariance = spatial_covariance(stft)  # (F, M, M)

        spectra = self.beamformer.compute_pseudo_spectrum(
            steering_vector, covariance
        )  # (F, T, P)

        spectra = spectra.mean(axis=0)  # (T, P)

        return PseudoSpectrumResult(grid, spectra)
