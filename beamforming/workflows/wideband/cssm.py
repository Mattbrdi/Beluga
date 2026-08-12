import numpy as np
from numpy.typing import NDArray


from beamforming.config import *

from beamforming.geometry import TetrahedralArray, Direction
from beamforming.grid import SearchGrid
from beamforming.results import PseudoSpectrumResult
from beamforming.signal.stft import STFTConfig
from beamforming.beamformers.base import Beamformer
from beamforming.workflows.wideband.basewideband import WidebandBeamformer

from beamforming.signal.covariance import spatial_covariance
from beamforming.steering import compute_steering_vector


class CSSM(WidebandBeamformer):
    def __init__(
        self,
        tetrahedra: TetrahedralArray,
        beamformer: Beamformer,
        stft_config: STFTConfig,
        reference_freq: float,
        initial_direction: Direction,
    ):
        super().__init__(tetrahedra, beamformer, stft_config)

        self.reference_freq = reference_freq
        self.initial_direction = initial_direction

    def compute_from_stft(
        self, stft: NDArray[np.complex128], freqs: NDArray[np.float64], grid: SearchGrid
    ) -> PseudoSpectrumResult:
        covariance = spatial_covariance(stft)  # (F, M, M)
        initial_grid = SearchGrid(
            self.initial_direction.theta, phi=self.initial_direction.phi
        )

        steering_f = compute_steering_vector(self.tetrahedra, initial_grid, freqs)[
            :, :, 0, 0
        ]  # (F, M)

        steering_ref = compute_steering_vector(
            self.tetrahedra, initial_grid, self.reference_freq
        )[
            0, :, 0, 0
        ]  # (M,)

        focus_matrixes = steering_ref[None, :] / steering_f  # (F,M)

        focused_covariances = np.mean(
            focus_matrixes[:, :, None]  # (F, M, 1)
            * covariance  # (F, M, M)
            * focus_matrixes.conj()[:, None, :],  # (F, 1, M)
            axis=0,
            keepdims=True
        )  # (1, M, M)

        steering_grid = compute_steering_vector(
            self.tetrahedra, grid, self.reference_freq
        )

        spectra = self.beamformer.compute_pseudo_spectrum(
            steering_grid, focused_covariances
        )

        return PseudoSpectrumResult(grid, spectra[0])
