"""Filter post VMD that remove frequencies. Input and output are AudioArrays"""

##### Computation imports #####
import numpy as np

##### Filters import #####
from scipy.signal import butter, sosfiltfilt


#### Homemade classes #####
from src.utils.sub_classes import AudioArray, AudioMetadata, PreFilterParameters

def butter_bp_filter(one_canal : np.ndarray, metadata:AudioMetadata, parameters : PreFilterParameters, frequency_range = None):
    # Default values : order 4
    if metadata.frequency_range is not None:
        filter_frequencies = metadata.frequency_range
    elif frequency_range is not None:
        filter_frequencies = frequency_range
    else:
        print('ERROR : Frequency range and metadata are None')
        return 
    
    sos = butter(parameters.order, [filter_frequencies[0], filter_frequencies[1]], btype="band", fs=metadata.sample_rate, output="sos")
    filtered_signal = sosfiltfilt(sos, one_canal)
    return filtered_signal

def filter_audio_array(audio_array : AudioArray, parameters : PreFilterParameters):
    if parameters.filtering_method is None:
        return audio_array
    for i in range(audio_array.data_array.shape[0]):
        audio_array.data_array[i] = butter_bp_filter(audio_array.data_array[i],audio_array.metadata,parameters)
    return(audio_array)

def filter_audio_array_from_calltype(audio_array : AudioArray, parameters : PreFilterParameters):
    if parameters.filtering_method is None:
        return audio_array
    if audio_array.metadata.beluga_call_type == 'Whistle':
        frequency_range = (500, 25000)
    else: 
        frequency_range = (25000, 190000)
    for i in range(audio_array.data_array.shape[0]):
        audio_array.data_array[i] = butter_bp_filter(audio_array.data_array[i],audio_array.metadata,parameters, frequency_range=frequency_range)
    return(audio_array)


def filter_data_array(data_array : np.ndarray, parameters : PreFilterParameters, sample_rate, call_type):
    frequency_range = {'Whistle': (500, 25000), 'Others': (25000, 190000)}
    if parameters.filtering_method is None:
        return data_array
    for i in range(data_array.shape[1]):
        data_array[:,i] = butter_bp_filter(data_array[:,i], AudioMetadata('', '', 0, 0, None, sample_rate, frequency_range[call_type],10000),parameters)
    return data_array