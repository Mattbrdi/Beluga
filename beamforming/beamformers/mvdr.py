import numpy as np
from numpy.typing import NDArray

from beamforming.beamformers.base import Beamformer

from beamforming.signal.covariance import diagonal_loading

class MVDR(Beamformer):
    def __init__(self):
        super().__init__()

    def compute_pseudo_spectrum(
        self,
        steering_vector: NDArray[np.complex128],
        correlation_matrix: NDArray[np.complex128],
    ) -> NDArray[np.float64]:
        """MVDR power computation from steering vector and correlation matrix

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

        if correlation_matrix.shape != (F, M, M):
            raise ValueError(
                "Expected correlation_matrix shape "
                f"{(F, M, M)}, got {correlation_matrix.shape}."
            )

        steering_flat = steering_vector.reshape(F, M, T * P)

        correlation_matrix = diagonal_loading(correlation_matrix)
        solved = np.linalg.solve(correlation_matrix, steering_flat)  # (F, M, T*P)

        denominator = np.sum(steering_flat.conj() * solved, axis=1).real
        denominator = np.maximum(denominator, 1e-12)
        power = 1.0 / denominator
        return power.reshape(F, T, P)
