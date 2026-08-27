import numpy as np
from numpy.typing import NDArray
import numpy.random as rd

from time_frequency_mask.config import STFTParamters
from time_frequency_mask.data_generation.io.data_parser import read_wav_file
from time_frequency_mask.data_generation.models.mask import Mask, WhistleMask
from time_frequency_mask.data_generation.models.audio_sample import (
    LabeledAudioSample,
    Whistle,
)


def generate_synthetic_whistle(
    central_frequency: float,
    duration: float,
    sampling_rate: int,
    total_duration: float = 1,
) -> tuple[NDArray[np.float64], Mask]:
    pass


def add_whistle_to_labeled_audio_sample(
    labeled_audio_sample: LabeledAudioSample, whistle: Whistle, start_time: float
) -> LabeledAudioSample:
    whistle.start_time = start_time
    return labeled_audio_sample + whistle


def generate_whistle_from_bank(
    wav_and_mask_dict: dict[str, np.ndarray],
    stft_parameters: STFTParamters,
    start_time: float,
) -> Whistle:
    num_available_whistles = len(wav_and_mask_dict["wav_paths"])

    whistle_index = rd.randint(0, num_available_whistles)
    wav_path, mask_path = (
        wav_and_mask_dict["wav_paths"][whistle_index],
        wav_and_mask_dict["mask_paths"][whistle_index],
    )

    waveform, sampling_rate = read_wav_file(wav_path, num_canals=1)

    waveform = waveform - np.mean(waveform)
    rms = np.sqrt((np.mean(waveform**2)))
    if rms != 0:
        waveform = waveform / rms

    mask = WhistleMask.from_path(mask_path, sampling_rate)
    return Whistle(waveform, mask, sampling_rate, stft_parameters, start_time)
