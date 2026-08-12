from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Beamformer(ABC):

    @abstractmethod
    def compute_pseudo_spectrum(
        self,
        steering_vector: NDArray[np.complex128],
        correlation_matrix: NDArray[np.complex128],
    ) -> NDArray[np.float64]:
        ...
