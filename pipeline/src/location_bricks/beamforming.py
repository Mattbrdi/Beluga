import sys
from pathlib import Path


import numpy as np
from numpy.typing import NDArray
from src.utils.sub_classes import Tetrahedra, Parameters
from beamforming.classes import TetrahedralArray
from beamforming.mathematics.beamformer import delay_and_sum_doa, mvdr_doa, music_doa
from beamforming.mathematics.wideband_beamforming import wideband_issm_mvdr_doa, wideband_issm_music_doa, wideband_cssm_mvdr_optimized_doa, wideband_cssm_music_optimized_doa
from beamforming.mathematics.masked_beamforming import masked_music_doa, masked_mvdr_doa
from time_frequency_mask.data_generation.models.mask import AudioMask
from time_frequency_mask.masknet.run_inference import get_mask_from_array, pad_crop_audio_array
from time_frequency_mask.tdoa_estimation.blob import output_blobs_from_mask, output_mask_from_blobs, blob_filtering_heuristic
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
    use_tf_mask = parameters.beamforming_parameters.use_tf_mask
    if not use_tf_mask:
        if beamformer_method == 'mvdr':
            return mvdr_doa(central_frequency, tetrahedra, audio_array)

        elif beamformer_method == 'music':
            return music_doa(central_frequency, tetrahedra, audio_array, num_expected_signals=1)

        elif beamformer_method == 'delay-and-sum':
            return delay_and_sum_doa(central_frequency, tetrahedra, audio_array)

        elif beamformer_method == "issm_mvdr":
            return wideband_issm_mvdr_doa(tetrahedra, audio_array)

        elif beamformer_method == "issm_music":
            return wideband_issm_music_doa(tetrahedra, audio_array)

        elif beamformer_method == "cssm_mvdr":
            return wideband_cssm_mvdr_optimized_doa(tetrahedra, audio_array, central_frequency)

        elif beamformer_method == "cssm_music":
            return wideband_cssm_music_optimized_doa(tetrahedra, audio_array, central_frequency)
        else:
            raise ValueError(f"Provided beamforming method doesn't exist or isn't implemented yet, got {beamformer_method}")
    else:
        audio_array_data = audio_array.copy()
        audio_array_data = pad_crop_audio_array(audio_array_data)
        original_mask = AudioMask(get_mask_from_array(audio_array_data, debug=False))

        blobs = output_blobs_from_mask(original_mask)
        blobs = blob_filtering_heuristic(blobs)
        filtered_mask = output_mask_from_blobs(blobs)

        if beamformer_method == 'mvdr':
            return masked_mvdr_doa(tetrahedra, audio_array_data, filtered_mask)

        elif beamformer_method == 'music':
            return masked_music_doa(tetrahedra, audio_array_data, filtered_mask)

        elif beamformer_method == "cssm_mvdr":
            return wideband_cssm_mvdr_optimized_doa(tetrahedra, audio_array_data, central_frequency, tf_mask=filtered_mask.data)

        elif beamformer_method == "cssm_music":
            return wideband_cssm_music_optimized_doa(tetrahedra, audio_array_data, central_frequency, tf_mask=filtered_mask.data)
        
        else:
            raise ValueError(f"Provided beamforming method doesn't exist or isn't implemented yet for masked_based beamforming, got {beamformer_method}")