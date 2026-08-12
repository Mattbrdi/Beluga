import numpy as np
from numpy.typing import NDArray

from beamforming.config import C

from beamforming.geometry import TetrahedralArray
from beamforming.grid import SearchGrid



def compute_steering_vector(
    tetrahedra: TetrahedralArray,
    search_grid: SearchGrid,
    freqs: float | NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Compute steering vector for provided angles and frequency

    Parameters
    ----------
    tetrahedra : TetrahedralArray
        geometry of the array points
    search_grid : SearchGrid
        range of search direction
    freqs : NDArray
        frequency of steering vector

    Returns
    -------
    NDArray[np.complex128]
        Steering vector array of size F, M, T, P, steering at each angle of the grid
    """

    freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))

    positions = tetrahedra.positions

    relative_positions = positions - positions[0]

    theta, phi = search_grid.mesh

    k = np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        dtype=np.float64,
    )  # (3, T, P)

    kdotp12 = np.tensordot(k, relative_positions[1], axes=(0, 0))  # (T, P)
    kdotp13 = np.tensordot(k, relative_positions[2], axes=(0, 0))  # (T, P)
    kdotp14 = np.tensordot(k, relative_positions[3], axes=(0, 0))  # (T, P)

    freq_view = freqs.reshape((freqs.size,) + (1,) * kdotp12.ndim)

    s = np.stack(
        [
            np.ones((freqs.shape[0], *kdotp12.shape), dtype=np.complex128),
            np.exp(2j * np.pi * freq_view * kdotp12 / C),
            np.exp(2j * np.pi * freq_view * kdotp13 / C),
            np.exp(2j * np.pi * freq_view * kdotp14 / C),
        ],
        axis=1,
    )
    return s  # (F, 4, T, P)
