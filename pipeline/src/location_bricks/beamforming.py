import sys
from pathlib import Path


import numpy as np
from numpy.typing import NDArray
from src.utils.sub_classes import Tetrahedra, Parameters
from beamforming.classes import TetrahedralArray
from beamforming.mathematics.beamformer import delay_and_sum_doa, mvdr_doa, music_doa
#TODO: Fix this by removing the need for sys and pathlib
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


#turn tetrahedra of pipeline to tetrahedra of beamforming
def get_tetrahedra(pipeline_tetrahedra : Tetrahedra) -> TetrahedralArray:
    """Convert a pipeline Tetrahedra into a TetrahedralArray for beamforming operations

    Parameters
    ----------
    pipeline_tetrahedra : Tetrahedra
        A tetrahedra instance of the Tetrahedra class in the pipeline

    Returns
    -------
    TetrahedralArray
        Array geometry in meters and in the enu referential
    """
    pos = pipeline_tetrahedra.rotated_hydro_pos_enu
    return TetrahedralArray(pos)

def beamforming_doa(parameters : Parameters, central_frequency : float, tetrahedra : TetrahedralArray, audio_array : NDArray[np.float64]):
    beamformer_method = parameters.beamforming_parameters.beamformer
    
    if beamformer_method == 'mvdr':
        return mvdr_doa(central_frequency, tetrahedra, audio_array)

    elif beamformer_method == 'music':
        return music_doa(central_frequency, tetrahedra, audio_array, num_expected_signals=1)

    elif beamformer_method == 'delay-and-sum':
        return delay_and_sum_doa(central_frequency, tetrahedra, audio_array)

    else:
        raise ValueError(f"Provided beamforming method doesn't exist or isn't implemented yet, got {beamformer_method}")
