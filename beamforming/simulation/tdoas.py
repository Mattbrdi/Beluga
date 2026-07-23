from beamforming.classes import *
from beamforming.configuration import *
from itertools import combinations


def get_tdoas_from_source(
    tetrahedra: Tetrahedra, source: Source
) -> NDArray[np.float64]:
    """Compute pairwise TDOA for tetrahedra hydrophones from provided source

    The pairwise TDOA at each hydrophone pairs of the tetrahedra are computed
    given a source three-dimensional position and the tetrahedra geometry.

    Parameters
    ----------
    tetrahedra : Tetrahedra
       Geometry of the four-hydrophone array. Positions are expected
       in metres.
    source : Source
        Source waveform and three-dimensional source position. The
        position is expected in meters.

    Returns
    -------
    NDArray[np.float64]
        An array containing the 6 pairwise TDOAs in seconds.

    Raises
    ------
    ValueError
        Cannot compute the TDOAs if the source and the tetrahedra are too close
    """
    tdoas = []

    direction_vector = tetrahedra.center - source.position

    if np.isclose(np.linalg.norm(direction_vector), 0):
        raise ValueError(
            f"source is too close to tetrahedra center. Source position is {source.position} and tetrahedra center position is {tetrahedra.center}"
        )

    direction_vector /= np.linalg.norm(direction_vector)

    for i, j in combinations(range(4), 2):
        p = tetrahedra.positions[j] - tetrahedra.positions[i]
        tdoas.append(np.dot(direction_vector, p) / C)

    return np.asarray(tdoas).astype(np.float64)
