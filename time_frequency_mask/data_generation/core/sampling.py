import numpy as np
import numpy.random as rd
from numpy.typing import NDArray

from scipy.signal import istft

from time_frequency_mask.configuration import MAX_TDOA, N_FREQS, MIN_FREQ, MAX_FREQ, SAMPLING_RATE, N_FFT, HOP_LENGTH, DURATION
from time_frequency_mask.stft import scipy_stft_complex
from time_frequency_mask.data_generation.core.preprocess import bandpass_filter
from time_frequency_mask.data_generation.generators.beluga_whistle_generator import generate_whistle_from_bank
from time_frequency_mask.data_generation.models.audio_sample import Whistle
from time_frequency_mask.data_generation.models.mask import WhistleMask

def set_num_whistles() -> int:
    # return rd.choice(a = 5, p = [0.2, 0.4, 0.3, 0.09, 0.01])
    return rd.choice(a = [1,2,3,4,5,6,7], p = [0.35, 0.25, 0.15, 0.12, 0.08, 0.03,0.02])

def set_start_time(duration : float) -> float:
    start_time = rd.uniform() * duration
    return start_time

def set_shift() -> float:
    return (2 * rd.random() - 1) * MAX_TDOA

def translate_waveform_in_frequency(waveform : NDArray[np.float64], shift : float) -> NDArray[np.float64]:

    if np.abs(shift) > MAX_FREQ - MIN_FREQ:
        raise ValueError(f"Incorrect shift frequency value provided got {shift} which is too large or too small compared to max value : {MAX_FREQ-MIN_FREQ}")

    freqs, times, D_full = scipy_stft_complex(waveform)

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

    _, waveform_translated = istft(D_full_translated, SAMPLING_RATE, 'hann', noverlap=N_FFT - HOP_LENGTH, nfft=N_FFT)
    if len(waveform) > len(waveform_translated):
        waveform_translated = np.pad(waveform_translated, (0, len(waveform)-len(waveform_translated)))
    if len(waveform_translated) > len(waveform):
        waveform_translated = waveform_translated[:len(waveform)]
    return bandpass_filter(waveform_translated, SAMPLING_RATE)

def translate_mask(mask : NDArray[np.uint8], shift : float) -> NDArray[np.uint8]:
    src_start = max(0, -shift)
    dst_start = max(0, shift)

    available_dst = N_FREQS - dst_start
    available_src = N_FREQS - src_start

    placed_width = min(available_dst, available_src)

    if placed_width <= 0:
        return np.zeros_like(mask)

    src_end = src_start + placed_width
    dst_end = dst_start + placed_width

    new_mask = np.zeros_like(mask)
    new_mask[dst_start:dst_end, :] = mask[src_start:src_end, :]
    return new_mask


def generate_whiste(start_time : float, is_augmentation : bool) -> Whistle:
    whistle = generate_whistle_from_bank(start_time)

    if is_augmentation:
        ys, xs = np.where(whistle.mask.data)

        min_y = ys.min()
        max_y = ys.max()

        counter = 0

        while counter < 3:
            shift = (2*rd.random() - 1) * (MAX_FREQ - MIN_FREQ)
            if abs(shift) <= 0.05 * (MAX_FREQ - MIN_FREQ):
                return whistle

            shift_idx = np.clip(int(np.round(N_FFT / SAMPLING_RATE * shift)),0, N_FREQS) 

            if max_y + shift_idx >= N_FREQS or min_y + shift_idx < 0:
                counter +=1
            
            else:
                waveform_translated = translate_waveform_in_frequency(whistle.waveform, shift_idx)

                translated_mask_data = translate_mask(whistle.mask.data, shift_idx)

                whistle = Whistle(waveform_translated, WhistleMask(translated_mask_data, SAMPLING_RATE), SAMPLING_RATE, start_time)

    return whistle

def set_snr() -> float:
    return 10.5*rd.random() - 0.5

def sample(is_augmentation = False, duration = DURATION) -> tuple[int, list[float], list[int], list[Whistle]]:
    """Sample values at each generated sample

    Returns
    -------
    tuple[int, list[float], list[int], list[Whistle]]
        return the number of whistle per sample, where they start, the hydrophones shifts and the generated whistles with their respective start_time
    """
    num_whistles = set_num_whistles()

    start_times = [set_start_time(duration) for _ in range(num_whistles)]

    shifts = [set_shift() for _ in range(3)]

    whistles = [generate_whiste(start_time, is_augmentation) for start_time in start_times]

    # snrs = [set_snr() for _ in range(4)]

    return num_whistles, start_times, shifts, whistles

def sample_impulsive_noise():
    pass