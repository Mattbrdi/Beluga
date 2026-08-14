import numpy as np
from numpy.typing import NDArray

from beamforming.beamformers.base import Beamformer


class MUSIC(Beamformer):
    def __init__(self, num_expected_sources):
        self.num_expected_sources = num_expected_sources
        super().__init__()

    def compute_pseudo_spectrum(
        self,
        steering_vector: NDArray[np.complex128],
        correlation_matrix: NDArray[np.complex128],
    ) -> NDArray[np.float64]:
        """MUSIC power computation from steering vector and correlation matrix

        Parameters
        ----------
        steering_vector : NDArray[np.complex128]
            Steering vector F, M, T, P sized matrix
        correlation_matrix : NDArray[np.complex128]
            correlation matrix F, M, M matrix used for MVDR algorithm

        Returns
        -------
        NDArray[np.float64]
            Power in db at each scanned angle
        """
        F, M, T, P = np.shape(steering_vector)
        S = self.num_expected_sources
        N = M - S

        if correlation_matrix.shape != (F, M, M):
            raise ValueError(
                "Expected correlation_matrix shape "
                f"{(F, M, M)}, got {correlation_matrix.shape}."
            )

        if not 0 <= S < M:
            raise ValueError(
                f"num_expected_sources must satisfy 0 <= S < M, " f"got S={S}, M={M}."
            )

        steering_flat = steering_vector.reshape(F, M, T * P)  # (F, M, G)
        _, eigvecs = np.linalg.eigh(correlation_matrix)  # # (F, M, M)

        noise_subspace = eigvecs[..., :N]  # Noise subspace (F, M, N)

        projected = noise_subspace.conj().swapaxes(-2, -1) @ steering_flat  # (F, M, G)

        denominator = np.sum(np.abs(projected) ** 2, axis=1).real  # (F, G)

        denominator = np.maximum(denominator, 1e-12)
        pseudo_spectrum = 1.0 / denominator
        return pseudo_spectrum.reshape(F, T, P)
