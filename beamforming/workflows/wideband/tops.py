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


class TOPS(WidebandBeamformer):
    def __init__(
        self,
        tetrahedra: TetrahedralArray,
        stft_config: STFTConfig,
        num_expected_sources: int,
        reference_frequency_index: int = None,
    ):
        super().__init__(tetrahedra, None, stft_config)

        self.num_expected_sources = num_expected_sources
        self.reference_frequency_index = reference_frequency_index

    def select_reference_frequency(self, eigenvalues, N) -> int:
        largest_noise = eigenvalues[:, N - 1] 
        smallest_signal = eigenvalues[:, N]

        eigengap = smallest_signal - largest_noise
        return int(np.argmax(eigengap))

    def compute_from_stft(
        self, stft: NDArray[np.complex128], freqs: NDArray[np.float64], grid: SearchGrid
    ) -> PseudoSpectrumResult:
        """Compute a TOPS pseudospectrum from multichannel STFT data.

        Parameters
        ----------
        stft:
            Multichannel STFT with shape (F, M, L).
        freqs:
            Frequency bins with shape (F,).
        grid:
            Search directions.

        Returns
        -------
        PseudoSpectrumResult
        TOPS pseudospectrum over the search grid. pseudospectrum with shape (T, P).
        """
        steering_vector = compute_steering_vector(self.tetrahedra, grid, freqs)
        correlation_matrix = spatial_covariance(stft)

        F, M, T, P = np.shape(steering_vector)
        S = self.num_expected_sources
        N = M - S

        if correlation_matrix.shape != (F, M, M):
            raise ValueError(
                "Expected correlation_matrix shape "
                f"{(F, M, M)}, got {correlation_matrix.shape}."
            )

        if not 1 <= S < M:
            raise ValueError(
                f"num_expected_sources must satisfy 1 <= S < M; " f"got S={S}, M={M}."
            )

        eigvals, eigvecs = np.linalg.eigh(correlation_matrix)  # (F, M, M)

        reference_index = self.reference_frequency_index

        if reference_index is None:
            reference_index = self.select_reference_frequency(eigvals, N)

        if not 0 <= reference_index < F:
            raise ValueError(
                "reference_frequency_index must be in "
                f"[0, {F}), got {reference_index}."
            )
        signal_subspace = eigvecs[reference_index, :, -S:]  # (M, S)
        noise_subspace = eigvecs[..., :N]  # (F, M, M - S)

        

        steering_norm = np.sum(np.abs(steering_vector) ** 2, axis=1)  # (F, T, P)

        s_ref = steering_vector[reference_index]  # (M, T, P)

        focus_ratio = steering_vector / s_ref[None, ...]  # (F, M, T, P)
        transformed_signal = (
            focus_ratio[:, :, None, :, :] * signal_subspace[None, :, :, None, None]
        )  # (F, M, S, T, P)

        projection_coeff = np.einsum(
            "fmtp,fmstp->fstp", steering_vector.conj(), transformed_signal
        )
        projection_coeff /= steering_norm[:, None, :, :]
        projected_signal = (
            transformed_signal
            - steering_vector[:, :, None, :, :] * projection_coeff[:, None, :, :, :]
        )  # (F, M, S, T, P)

        other_freqs = np.arange(F) != reference_index

        projected_signal = projected_signal[other_freqs]
        noise_subspace = noise_subspace[other_freqs]

        D = np.einsum(
            "fmstp,fmn->fsntp", projected_signal.conj(), noise_subspace
        )  # (F-1, S, N, T, P)
        D = D.transpose(1, 0, 2, 3, 4).reshape(
            S, (F - 1) * N, T, P
        )  # (S, (F-1) * N, T, P)
        D_batch = D.transpose(2, 3, 0, 1)  # (T, P, S, (F-1) * N)

        singular_values = np.linalg.svdvals(D_batch)  # (T, P, S)
        sigma_min = singular_values[..., -1]  # (T, P)

        pseudo_spectrum = 1.0 / np.maximum(sigma_min, 1e-12)

        return PseudoSpectrumResult(grid, pseudo_spectrum)
