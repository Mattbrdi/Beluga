from typing import Optional, List
import numpy as np
from itertools import combinations


#TODO : refactor pour ne pas copier coller deux fois le code de fusion_bf et high_fusion
def fusion_bf(wave_vectors_list: List[np.ndarray], wave_vector_error_variance: List[np.ndarray], environment, projection_plan: Optional[float]):
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
    return position.reshape(-1, 1), estimated_error


def direction_vector_from_tdoas(
    tdoas_measured: np.ndarray,
    tetrahedra,
    tdoas_mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Estimate a normalized source direction in ENU from one tetrahedron.

    The minus sign follows the TDOA convention used by ``tdoa_from_pair`` and
    makes the resulting vector point from the tetrahedron towards the source.
    Only pairs selected by ``tdoas_mask`` participate in the least-squares
    estimate.
    """
    tdoas_measured = np.asarray(tdoas_measured, dtype=float).reshape(-1)
    v_matrix = np.asarray(tetrahedra.v_matrix, dtype=float)
    if len(tdoas_measured) != len(v_matrix):
        raise ValueError(
            "TDOA vector and tetrahedron V matrix must have the same length: "
            f"{len(tdoas_measured)} != {len(v_matrix)}"
        )

    if tdoas_mask is None:
        mask = np.ones(len(tdoas_measured), dtype=bool)
    else:
        mask = np.asarray(tdoas_mask, dtype=bool).reshape(-1)
        if len(mask) != len(tdoas_measured):
            raise ValueError("TDOA mask and TDOA vector must have the same length.")

    mask &= np.isfinite(tdoas_measured)
    selected_v_matrix = v_matrix[mask]
    selected_tdoas = tdoas_measured[mask]
    dimensions = v_matrix.shape[1]
    if len(selected_tdoas) < dimensions:
        return None
    if np.linalg.matrix_rank(selected_v_matrix) < dimensions:
        return None

    direction = -np.linalg.pinv(selected_v_matrix) @ selected_tdoas.reshape(-1, 1)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        return None
    return direction / norm

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
            normalized_wave_vector = direction_vector_from_tdoas(tdoa, tetrahedra)
            if normalized_wave_vector is None:
                normalized_wave_vector = np.full(
                    (tetrahedra.v_matrix.shape[1], 1),
                    np.nan,
                )
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

def two_tetra_intersection(
    wave_vector_pair,
    origins_enu_pair,
    projection_plan: Optional[float],
    project_directions_to_xy: bool = False,
):
    """
    Calculate the intersection point of two tetrahedra.

    Parameters:
    - wave_vector_pair: Pair of wave vectors.
    - origins_enu_pair: Pair of origins in ENU coordinates.
    - projection_plan: Optional Z value returned when XY projection is used.
    - project_directions_to_xy: If True, project 3D directions to XY before
      intersection. 2D wave vectors are always intersected in XY.

    Returns:
    - position: Intersection position or NaN if vectors are divergent.
    """
    vector_dimension = wave_vector_pair[0].shape[0]
    project_to_xy = project_directions_to_xy or vector_dimension == 2

    if project_to_xy:
        # Les vecteurs peuvent avoir ete estimes en 3D avec les 6 TDOA. Pour
        # la triangulation, on ne conserve ici que leur projection horizontale.
        direction_one = wave_vector_pair[0].ravel()[:2]
        direction_two = wave_vector_pair[1].ravel()[:2]
        origin_one = np.asarray(origins_enu_pair[0], dtype=float)[:2]
        origin_two = np.asarray(origins_enu_pair[1], dtype=float)[:2]
    else:
        direction_one = wave_vector_pair[0].ravel()
        direction_two = wave_vector_pair[1].ravel()
        origin_one = np.asarray(origins_enu_pair[0], dtype=float)
        origin_two = np.asarray(origins_enu_pair[1], dtype=float)

    direction_matrix = np.column_stack((direction_one, -direction_two))
    origins_difference = origin_two - origin_one
    optimal_weights = np.linalg.lstsq(direction_matrix, origins_difference, rcond=None)[0]
    """if np.sign(optimal_weights[0])!=np.sign(optimal_weights[1]):
        #TODO Check if relevant
        print("Vecteurs d'onde divergents")
        return np.full(3, np.nan)"""
    if not project_to_xy:
        position = 0.5*(
            origin_one + optimal_weights[0] * direction_one
            + origin_two + optimal_weights[1] * direction_two
            )
        return position

    position_xy = 0.5*(
        origin_one + optimal_weights[0] * direction_one
        + origin_two + optimal_weights[1] * direction_two
        )
    if projection_plan is None:
        projection_plan = np.nan
    position = np.array([position_xy[0], position_xy[1], projection_plan])
    return position

def high_fusion(
    tdoas_measured: List[np.ndarray],
    tdoas_error_variance: List[np.ndarray],
    environment,
    projection_plan: Optional[float],
    project_directions_to_xy: bool = False,
):
    """
    Perform high-level fusion of audio arrays and TDOAs to estimate position.

    Parameters:
    - audio_arrays: List of audio arrays.
    - tdoas_measured: List of measured TDOA arrays.
    - tdoas_error_variance: List of TDOA error variances.
    - environment: Environment containing tetrahedra information.
    - projection_plan: Optional Z value returned when XY projection is used.
    - project_directions_to_xy: If True, project 3D wave vectors to XY before
      intersecting them.

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
        positions.append(
            two_tetra_intersection(
                wave_pair,
                origins,
                projection_plan,
                project_directions_to_xy=project_directions_to_xy,
            )
        )
    positions = np.array(positions)
    position = np.mean(positions, axis=0)
    estimated_error = np.zeros(3) # TODO ajouter module d'erreur ici si on garde
    return position, estimated_error

def main():
    # tdoas_measured =  [
    #     [-0.0001796875, -2.604166666666667e-05, -0.00011458333333333333, 0.00015364583333333333, 0.00019010416666666665, -9.635416666666667e-05]
    # ]

    # tdoas_measured =  [
    #     [0.0001796875, 2.604166666666667e-05, 0.00011458333333333333, -0.00015364583333333333, -0.00019010416666666665, 9.635416666666667e-05]
    # ]

    tdoas_measured = [

    ]
    from pathlib import Path
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.utils.sub_classes import Parameters, Environment
    from data_position_stats import enu_to_lla
    param_path = 'jsons/parameters/default_parameters.json'

    env_path = 'jsons/environments/env_cacouna_may2026.json'

    parameters = Parameters(param_path)
    environment = Environment(env_path, parameters.location_parameters.use_h4)
    
    wave_vectors_list = wave_vectors(tdoas_measured, environment)
    wave_vectors_list = np.array(wave_vectors_list)
    print(wave_vectors_list, enu_to_lla(*(10*wave_vectors_list[0]), *[47.93961, -69.52256, 0]))
if __name__ == "__main__":
    main()