import numpy as np 
from numpy.typing import NDArray

from scipy.signal import butter, sosfilt

from time_frequency_mask.configuration import SAMPLING_RATE, MIN_FREQ, MAX_FREQ
from time_frequency_mask.data_generation.models.audio_sample import TetrahedraAudioSample

def lowpass_filter(waveform : NDArray[np.float64], sampling_rate : float = SAMPLING_RATE, high : float = MAX_FREQ) -> NDArray[np.float64]:
    sos = butter(4, high, 'sos', fs=sampling_rate)
    return sosfilt(sos, waveform)

def bandpass_filter(waveform : NDArray[np.float64], sampling_rate : float = SAMPLING_RATE, low : float = MIN_FREQ, high : float = MAX_FREQ) -> NDArray[np.float64]:
    sos = butter(4, (low, high), 'band', fs = sampling_rate, output="sos")
    return sosfilt(sos, waveform)

def waveform_rescale(waveform : NDArray[np.float64]):
    pass

def preprocess(list_of_tetrahedra_audio_sample : list[TetrahedraAudioSample]):
    rescales = []
    for channel_idx in range(4):
        concat_array = np.concatenate([sample.shifted_waveforms[channel_idx] for sample in list_of_tetrahedra_audio_sample])
        
        rescale = np.percentile(np.abs(concat_array), 95)
        if rescale == 0:
            rescale = 1
        rescales.append(rescale)

    for sample in list_of_tetrahedra_audio_sample:
        for channel_idx in range(4):
            sample.shifted_waveforms[channel_idx] /= rescales[channel_idx]
            sample.shifted_waveforms[channel_idx] = np.clip(sample.shifted_waveforms[channel_idx], -5, 5)

def smoothing_butterworth(waveform: NDArray[np.float64]) -> NDArray[np.float64]:
    return lowpass_filter(waveform)