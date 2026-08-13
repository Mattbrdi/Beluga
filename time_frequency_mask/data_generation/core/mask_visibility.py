import numpy as np
import numpy.random as rd
from numpy.typing import NDArray

from time_frequency_mask.configuration import *
from time_frequency_mask.data_generation.core.power_computation import compute_P_in_t, compute_P_moy
from time_frequency_mask.stft import scipy_stft_complex_psd, frequency_band
BOX_SIZE_TIMES = 10
BOX_SIZE_FREQS = 10

df = SAMPLING_RATE / N_FFT

def get_idx(freq_idx : int, time_idx : int, n_freqs : int, n_times : int, box_size_freqs = BOX_SIZE_FREQS, box_size_times = BOX_SIZE_TIMES) -> tuple[int, int, int, int]:
    start_idx_freqs = freq_idx * box_size_freqs
    end_idx_freqs = (freq_idx+1) * box_size_freqs

    start_idx_times = time_idx * box_size_times
    end_idx_times = (time_idx+1) * box_size_times

    return min(start_idx_freqs, n_freqs), min(end_idx_freqs, n_freqs), min(start_idx_times, n_times), min(end_idx_times, n_times)

def compute_signal_bin_mask_discrimination(Zxx_s_nu : NDArray[np.complex128], Zxx_nu : NDArray[np.complex128], mask : NDArray[np.uint8], box_size_freqs = BOX_SIZE_FREQS, box_size_times = BOX_SIZE_TIMES, perc_thr : float = 0.5):

    if Zxx_s_nu.shape != Zxx_nu.shape:
        raise ValueError(
            f"Incorrect shape for Zxx_s_nu and Zxx_nu got {Zxx_s_nu.shape} and {Zxx_nu.shape}"
        )
    n_freqs, n_times = Zxx_s_nu.shape

    time_count = int(np.ceil(n_times / box_size_times)) # num of boxes in the horizontal axis
    freq_count = int(np.ceil(n_freqs / box_size_freqs)) # num of boxes in the vertical axis

    true_or_false_board = np.zeros(shape=(freq_count, time_count), dtype=bool)
    
    Pxx_s_nu = np.abs(Zxx_s_nu)**2
    Pxx_nu = np.abs(Zxx_nu)**2
    global_noise_psd = np.sum(2 * df * Pxx_s_nu[mask == 0]) / np.sum(2 * df * (mask == 0))

    #TODO: Vectorize and remove for loops
    for freq_idx in range(freq_count):
        for time_idx in range(time_count):
            start_i, end_i, start_j, end_j = get_idx(freq_idx, time_idx, n_freqs, n_times)

            Pxx_s_nu_ij = Pxx_s_nu[start_i:end_i, start_j:end_j]
            Pxx_nu_ij = Pxx_nu[start_i:end_i, start_j:end_j]
            mask_ij = mask[start_i:end_i, start_j:end_j]

            awgn_mean_power = np.mean(compute_P_in_t(Pxx_nu_ij, mask_ij))
            signal_mean_power, noise_mean_power = compute_P_moy(Pxx_s_nu_ij, mask_ij, global_noise_psd)
            
            true_or_false_board[freq_idx, time_idx] = signal_mean_power > perc_thr * (awgn_mean_power + noise_mean_power)
    return true_or_false_board

# def update_mask_for_noise(true_or_false_board : NDArray[np.uint8], mask : NDArray[np.uint8]):
    

def update_mask_for_noise(waveform : NDArray[np.float64], noise : NDArray[np.float64], mask : NDArray[np.uint8]) -> NDArray[np.uint8]:
    if not np.any(mask !=0):
        return mask
    
    freqs, _, Zxx_s_nu = scipy_stft_complex_psd(waveform)
    freqs, Zxx_s_nu = frequency_band(freqs, Zxx_s_nu)

    freqs, _, Zxx_nu = scipy_stft_complex_psd(noise)
    freqs, Zxx_nu = frequency_band(freqs, Zxx_nu)

    n_freqs, n_times = Zxx_s_nu.shape
    
    true_or_false_board = compute_signal_bin_mask_discrimination(Zxx_s_nu, Zxx_nu, mask)
    for i, line in enumerate(true_or_false_board):
        for j, boolean in enumerate(line):
            if not boolean:
                start_i, end_i, start_j, end_j = get_idx(i, j, n_freqs, n_times)
                mask[start_i:end_i, start_j:end_j] = False
    return mask