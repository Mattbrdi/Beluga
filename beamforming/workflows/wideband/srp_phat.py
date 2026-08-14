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
from beamforming.signal.stft import compute_band_stft
from beamforming.steering import compute_steering_vector



class SRPPHAT(WidebandBeamformer):
    def __init__(
        self,
        tetrahedra: TetrahedralArray,
        stft_config: STFTConfig,
    ):
        super().__init__(tetrahedra, None, stft_config)

    def compute_weighted(
        self,
        signal: NDArray[np.float64],
        grid: SearchGrid,
        tf_weights: NDArray[np.uint8],
    ) -> PseudoSpectrumResult:
        freqs, _, Zxx = compute_band_stft(signal, self.stft_config, SAMPLING_RATE)
        return self.compute_weighted_from_stft(Zxx, freqs, grid, tf_weights)

    def compute_from_stft(
        self, stft: NDArray[np.complex128], freqs: NDArray[np.float64], grid: SearchGrid
    ) -> PseudoSpectrumResult:
        F, _, Ti = stft.shape
        tf_weights = np.ones(shape=(F, Ti), dtype=np.uint8)
        return self.compute_weighted_from_stft(stft, freqs, grid, tf_weights)

    def compute_weighted_from_stft(
        self,
        stft: NDArray[np.complex128],
        freqs: NDArray[np.float64],
        grid: SearchGrid,
        tf_weights,
    ) -> PseudoSpectrumResult:
        """Compute the SRP-PHAT spatial response.

        Parameters
        ----------
        steering_vector:
            Steering vectors with shape (F, M, T, P).
        stft:
            Multichannel STFT with shape (F, M, L).
        tf_weights:
            Time-frequency weights with shape (F, L).

        Returns
        -------
        NDArray[np.float64]
            SRP-PHAT spatial response with shape (T, P).
        """
        steering_vector = compute_steering_vector(self.tetrahedra, grid, freqs)
        F, M, T, P = steering_vector.shape

        if stft.ndim != 3:
            raise ValueError(f"Expected STFT shape (F, M, L), got {stft.shape}.")

        if stft.shape[:2] != (F, M):
            raise ValueError(
                f"Expected STFT first dimensions {(F, M)}, " f"got {stft.shape[:2]}."
            )

        n_frames = stft.shape[2]

        if tf_weights.shape != (F, n_frames):
            raise ValueError(
                f"Expected tf_weights shape {(F, n_frames)}, "
                f"got {tf_weights.shape}."
            )

        i, j = np.triu_indices(M, k=1)
        Q = len(i)
        cross_ftq = (stft[:, i, :] * stft[:, j, :].conj()).transpose(
            0, 2, 1
        )  # (F, Ti, Q)

        cross_phat_ftq = np.divide(
            cross_ftq,
            np.abs(cross_ftq),
            out=np.zeros_like(cross_ftq),
            where=np.abs(cross_ftq) > 1e-12,
        )

        tf_weights_sum = tf_weights.sum(axis=1)  # (F,)

        cross_phat_fq = np.einsum(
            "ftq,ft->fq",
            cross_phat_ftq,
            tf_weights,
            optimize=True,
        )  # (F, Q)

        cross_phat_fq = np.divide(
            cross_phat_fq,
            tf_weights_sum[:, None],
            out=np.zeros_like(cross_phat_fq),
            where=tf_weights_sum[:, None] > 0,
        )
        # (F, Q)

        pair_steering = (
            steering_vector[:, i, :, :] * steering_vector[:, j, :, :].conj()
        )  #  # (F, Q, T, P)

        srp_phat = np.real(
            np.einsum("fq,fqtp->tp", cross_phat_fq, pair_steering.conj(), optimize=True)
        )

        return PseudoSpectrumResult(grid, srp_phat)
