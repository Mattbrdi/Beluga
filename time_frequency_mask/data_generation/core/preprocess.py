import numpy as np
from numpy.typing import NDArray

from scipy.signal import butter, sosfilt

from time_frequency_mask.config import AudioParameters
from time_frequency_mask.data_generation.models.audio_sample import (
    TetrahedraAudioSample,
)


def lowpass_filter(
    waveform: NDArray[np.float64], audio_parameters: AudioParameters
) -> NDArray[np.float64]:
    sos = butter(4, audio_parameters.max_freq, "sos", fs=audio_parameters.sampling_rate)
    return sosfilt(sos, waveform)


def bandpass_filter(
    waveform: NDArray[np.float64], sampling_rate: float, low: float, high: float
) -> NDArray[np.float64]:
    sos = butter(4, (low, high), "band", fs=sampling_rate, output="sos")
    return sosfilt(sos, waveform)


def waveform_rescale(waveform: NDArray[np.float64]):
    pass


def preprocess(list_of_tetrahedra_audio_sample: list[TetrahedraAudioSample]):
    rescales = []
    M = list_of_tetrahedra_audio_sample[0].shifted_waveforms.shape[0]
    for channel_idx in range(M):
        concat_array = np.concatenate(
            [
                sample.shifted_waveforms[channel_idx]
                for sample in list_of_tetrahedra_audio_sample
            ]
        )

        rescale = np.percentile(np.abs(concat_array), 95)
        if rescale == 0:
            rescale = 1
        rescales.append(rescale)

    for sample in list_of_tetrahedra_audio_sample:
        for channel_idx in range(M):
            sample.shifted_waveforms[channel_idx] /= rescales[channel_idx]
            sample.shifted_waveforms[channel_idx] = np.clip(
                sample.shifted_waveforms[channel_idx], -5, 5
            )


def smoothing_butterworth(
    waveform: NDArray[np.float64], audio_parameters: AudioParameters
) -> NDArray[np.float64]:
    return lowpass_filter(waveform, audio_parameters)
