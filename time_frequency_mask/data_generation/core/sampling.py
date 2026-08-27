import numpy as np
import numpy.random as rd
from numpy.typing import NDArray

from scipy.signal import istft
from time_frequency_mask.config import *
from time_frequency_mask.stft import scipy_stft_complex
from time_frequency_mask.data_generation.core.preprocess import bandpass_filter
from time_frequency_mask.data_generation.generators.beluga_whistle_generator import (
    generate_whistle_from_bank,
)
from time_frequency_mask.data_generation.models.audio_sample import Whistle
from time_frequency_mask.data_generation.models.mask import WhistleMask


def set_num_whistles(max_num_whistles) -> int:
    # return rd.choice(a = 5, p = [0.2, 0.4, 0.3, 0.09, 0.01])
    return rd.choice(a=np.arange(max_num_whistles))


def set_start_time(duration: float) -> float:
    start_time = rd.uniform() * duration
    return start_time


def translate_waveform_in_frequency(
    waveform: NDArray[np.float64], parameters: Parameters, shift: int
) -> NDArray[np.float64]:
    sampling_rate, min_freq, max_freq = (
        parameters.audio.sampling_rate,
        parameters.audio.min_freq,
        parameters.audio.max_freq,
    )
    n_fft, start_index = parameters.stft.n_fft, parameters.stft.freq_index(
        sampling_rate, min_freq
    )

    if (
        np.abs(
            (shift + parameters.stft.freq_index(sampling_rate, start_index))
            * sampling_rate
            / n_fft
        )
        > max_freq - min_freq
    ):
        raise ValueError(
            f"Incorrect shift frequency value provided got {shift} which is too large or too small compared to max value : {max_freq-min_freq}"
        )

    freqs, times, D_full = scipy_stft_complex(
        waveform, parameters.audio.sampling_rate, parameters.stft
    )

    F, T = D_full.shape

    src_start = max(0, -shift)
    dst_start = max(0, shift)

    available_dst = F - dst_start
    available_src = F - src_start

    placed_width = min(available_dst, available_src)

    if placed_width <= 0:
        print("debug")
        return np.zeros_like(waveform)

    src_end = src_start + placed_width
    dst_end = dst_start + placed_width

    D_full_translated = np.zeros_like(D_full)

    D_full_translated[dst_start:dst_end] = D_full[src_start:src_end]

    _, waveform_translated = istft(
        D_full_translated,
        parameters.audio.sampling_rate,
        window=parameters.stft.window,
        noverlap=parameters.stft.n_fft - parameters.stft.hop_length,
        nfft=parameters.stft.n_fft,
    )
    if len(waveform) > len(waveform_translated):
        waveform_translated = np.pad(
            waveform_translated, (0, len(waveform) - len(waveform_translated))
        )
    if len(waveform_translated) > len(waveform):
        waveform_translated = waveform_translated[: len(waveform)]
    return bandpass_filter(
        waveform_translated,
        parameters.audio.sampling_rate,
        parameters.audio.min_freq,
        parameters.audio.max_freq,
    )


def translate_mask(mask: NDArray[np.uint8], shift: float) -> NDArray[np.uint8]:
    src_start = max(0, -shift)
    dst_start = max(0, shift)

    available_dst = mask.shape[0] - dst_start
    available_src = mask.shape[0] - src_start

    placed_width = min(available_dst, available_src)

    if placed_width <= 0:
        return np.zeros_like(mask)

    src_end = src_start + placed_width
    dst_end = dst_start + placed_width

    new_mask = np.zeros_like(mask)
    new_mask[dst_start:dst_end, :] = mask[src_start:src_end, :]
    return new_mask


def generate_whiste(
    wav_and_mask_dict: dict[str, np.ndarray],
    start_time: float,
    parameters: Parameters,
    is_augmentation: bool = False,
) -> Whistle:

    whistle = generate_whistle_from_bank(wav_and_mask_dict, parameters.stft, start_time)

    if is_augmentation:
        audio, stft = parameters.audio, parameters.stft
        n_freqs = stft.num_frequency_bins_between(
            audio.sampling_rate, audio.min_freq, audio.max_freq
        )

        ys, xs = np.where(whistle.mask.data)

        min_y = ys.min()
        max_y = ys.max()

        counter = 0

        while counter < 3:
            shift = (2 * rd.random() - 1) * (audio.max_freq - audio.min_freq)
            if abs(shift) <= 0.05 * (audio.max_freq - audio.min_freq):
                return whistle

            shift_idx = np.clip(
                int(np.round(stft.n_fft / audio.sampling_rate * shift)), 0, n_freqs
            )

            if max_y + shift_idx >= n_freqs or min_y + shift_idx < 0:
                counter += 1

            else:
                waveform_translated = translate_waveform_in_frequency(
                    whistle.waveform, parameters, shift_idx
                )

                translated_mask_data = translate_mask(whistle.mask.data, shift_idx)

                whistle = Whistle(
                    waveform_translated,
                    WhistleMask(translated_mask_data, audio.sampling_rate),
                    audio.sampling_rate,
                    parameters.stft,
                    start_time,
                )

    return whistle


def set_snr() -> float:
    return 10.5 * rd.random() - 0.5


def sample_duration(audio_parameters: AudioParameters):
    if audio_parameters.duration:
        return audio_parameters.duration
    else:
        return audio_parameters.min_duration + rd.random() * (
            audio_parameters.max_duration - audio_parameters.min_duration
        )


def sample_whistles(
    whistle_bank_paths: dict[str, np.ndarray], parameters: Parameters, duration: float
) -> list[Whistle]:
    """Sample values at each generated sample

    Returns
    -------
    tuple[int, list[float], list[int], list[Whistle]]
        return the number of whistle per sample, where they start, the hydrophones shifts and the generated whistles with their respective start_time
    """
    num_whistles = set_num_whistles(parameters.generation.max_num_whistles)

    start_times = [set_start_time(duration) for _ in range(num_whistles)]

    whistles = [
        generate_whiste(whistle_bank_paths, start_time, parameters, False)
        for start_time in start_times
    ]

    # snrs = [set_snr() for _ in range(4)]

    return whistles


def sample_shifts(parameters: Parameters):
    sample_direction = rd.normal(size=3)
    sample_direction /= np.linalg.norm(sample_direction)
    array_geometry = parameters.array.array_geometry
    p = array_geometry - array_geometry[0]
    tdoas = p @ sample_direction / parameters.array.sound_speed
    return tdoas[1:]


def sample_snrs(parameters: Parameters):
    snr = rd.uniform(low=parameters.noise.min_snr, high=parameters.noise.max_snr)
    return [
        rd.normal(loc=snr, scale=parameters.noise.snr_variance)
        for _ in range(parameters.array.num_mics)
    ]


def sample_impulsive_noise():
    raise NotImplementedError("Impulsive noise is not yet implemented")
