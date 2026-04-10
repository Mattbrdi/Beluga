from typing import Optional, List
import numpy as np
from itertools import combinations

def wave_vectors(tdoas_measured: List[np.ndarray], environment):
    """
    Calculate wave vectors from measured TDOAs and environment tetrahedra.

    Parameters:
    - tdoas_measured: List of measured TDOA arrays.
    - environment: Environment containing tetrahedra information.

    Returns:
    - wave_vectors_list: List of normalized wave vectors.
    """
    wave_vectors_list = []
    for tdoa, tetrahedra in zip(tdoas_measured, environment.tetrahedras.values()):
        if tetrahedra.is_active:
            wave_vector = -np.linalg.pinv(tetrahedra.v_matrix) @ tdoa.reshape(-1,1)
            normalized_wave_vector = wave_vector / np.linalg.norm(wave_vector)
            # Expected shape : 2 x 1 for 2D, 3 x 1 for 3D. The shape of v_matrix determines 2D or 3D
            wave_vectors_list.append(normalized_wave_vector)
    return wave_vectors_list

def azimuths(wave_vectors: List[np.ndarray]):
    """
    Calculate azimuths from wave vectors.

    Parameters:
    - wave_vectors: List of wave vectors.

    Returns:
    - azimuths: List of azimuths in radians.
    """
    azimuths = []
    for wave_vector in wave_vectors:
        azimuths.append(np.arctan2(wave_vector[1], wave_vector[0]))
    return azimuths

def elevations(wave_vectors: List[np.ndarray]):
    """
    Calculate elevations from wave vectors if working in 3D.

    Parameters:
    - wave_vectors: List of wave vectors.

    Returns:
    - elevations: List of elevations in radians or NaN for 2D.
    """
    elevations = []
    for wave_vector in wave_vectors:
        if wave_vectors[0].shape == (3, 1):
            elevations.append(np.arctan(wave_vector[2] / np.sqrt(wave_vector[0]**2 + wave_vector[1]**2)))
        else:
            elevations.append(np.nan) # 2D processing
    return elevations

def two_tetra_intersection(wave_vector_pair, origins_enu_pair, projection_plan: Optional[float]):
    """
    Calculate the intersection point of two tetrahedra.

    Parameters:
    - wave_vector_pair: Pair of wave vectors.
    - origins_enu_pair: Pair of origins in ENU coordinates.
    - projection_plan: Optional projection plane for 2D processing.

    Returns:
    - position: Intersection position or NaN if vectors are divergent.
    """
    direction_matrix = np.array([wave_vector_pair[0], -wave_vector_pair[1]]).T
    # TODO maybe np concat is better
    if wave_vector_pair[0].shape == (3, 1):
        origins_difference = origins_enu_pair[1] - origins_enu_pair[0]
    else:
        origins_difference = origins_enu_pair[1][:2] - origins_enu_pair[0][:2]
    direction_matrix = direction_matrix[0, :, :]
    optimal_weights = np.linalg.lstsq(direction_matrix, origins_difference, rcond=None)[0]
    """if np.sign(optimal_weights[0])!=np.sign(optimal_weights[1]):
        #TODO Check if relevant
        print("Vecteurs d'onde divergents")
        return np.full(3, np.nan)"""
    if wave_vector_pair[0].shape == (3, 1):
        position = 0.5*(
            origins_enu_pair[0] + optimal_weights[0] * wave_vector_pair[0].ravel()
            + origins_enu_pair[1] + optimal_weights[1] * wave_vector_pair[1].ravel()
            )
        return position

    position_xy = 0.5*(
        origins_enu_pair[0][:2] + optimal_weights[0] * wave_vector_pair[0].ravel()
        + origins_enu_pair[1][:2] + optimal_weights[1] * wave_vector_pair[1].ravel()
        )
    if projection_plan is None:
        projection_plan = np.nan
    position = np.array([position_xy[0], position_xy[1], projection_plan])
    return position

def high_fusion(tdoas_measured: List[np.ndarray], tdoas_error_variance: List[np.ndarray], environment, projection_plan: Optional[float]):
    """
    Perform high-level fusion of audio arrays and TDOAs to estimate position.

    Parameters:
    - audio_arrays: List of audio arrays.
    - tdoas_measured: List of measured TDOA arrays.
    - tdoas_error_variance: List of TDOA error variances.
    - environment: Environment containing tetrahedra information.
    - projection_plan: Optional projection plane for 2D processing.

    Returns:
    - position: Estimated position.
    - estimated_error: Estimated error.
    """
    wave_vectors_list = wave_vectors(tdoas_measured, environment)
    tetrahedras_origins_enu = [np.array(tetra.origin_enu) for tetra in environment.tetrahedras.values() if tetra.is_active]
    if len(wave_vectors_list) <= 1:
        print(f'WARNING : Only one active tetrahedra, returning NaNs')
        return np.full((3, 1), np.nan), np.full((3, 1), np.nan)
    positions = []
    for wave_pair, origins in zip(combinations(wave_vectors_list, 2), combinations(tetrahedras_origins_enu, 2)):
        positions.append(two_tetra_intersection(wave_pair, origins, projection_plan))
    positions = np.array(positions)
    position = np.mean(positions, axis=0)
    estimated_error = np.zeros(3) # TODO ajouter module d'erreur ici si on garde
    return position, estimated_error