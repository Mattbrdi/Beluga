import numpy as np
import numpy.random as rd
from numpy.typing import NDArray

from time_frequency_mask.configuration import *
from time_frequency_mask.stft import scipy_spectrogram, frequency_band, scipy_stft_complex, scipy_stft_complex_psd

df = SAMPLING_RATE / N_FFT

def compute_mean_power(waveform : NDArray[np.float64], mask = None) -> float:
    freqs, time, Zxx = scipy_stft_complex_psd(waveform)
    freqs, Zxx = frequency_band(freqs, Zxx)
    Pxx = np.abs(Zxx)**2

    if mask is None:
        mask = np.ones_like(Pxx)

    return np.mean(np.sum(2*df*Pxx * mask, axis = 0))

def compute_P_in_t(Pxx : NDArray[np.float64], mask : NDArray[np.uint8]) -> NDArray[np.float64]: 
    if Pxx.shape != mask.shape:
        raise ValueError(f"Incorrect shape got {Pxx.shape} for Zxx and {mask.shape} for mask")
    
    M = mask !=0
    return np.sum(2 * df * Pxx * M, axis=0)

def compute_noise_power_t(Pxx : NDArray[np.float64], mask : NDArray[np.uint8], fallback_noise_psd) -> NDArray[np.float64]: 
    if Pxx.shape != mask.shape:
        raise ValueError(f"Incorrect shape got {Pxx.shape} for Zxx and {mask.shape} for mask")

    M_in = mask != 0
    M_out = mask == 0

    P_out_t = 2 * df * np.sum(Pxx * M_out, axis=0)
    B_out_t = np.sum(M_out*2*df, axis = 0)
    B_in_t = np.sum(M_in*2*df, axis=0)

    valid = B_out_t > 0

    if not np.any(valid):
        return fallback_noise_psd * B_in_t

    noise_density_mean = np.sum(P_out_t[valid]) / np.sum(B_out_t[valid])

    noise_in_M_t = noise_density_mean * B_in_t

    return noise_in_M_t

def compute_P_moy(Pxx_ij : NDArray[np.float64], mask_ij : NDArray[np.uint8], fallback_noise_psd : float) -> tuple[float, float]:
    M = mask_ij != 0

    if np.sum(M) == 0:
        return 0.0, 0.0

    P_signal_and_noise_in_bin_t = compute_P_in_t(Pxx_ij, mask_ij)
    P_noise_in_bin_t = compute_noise_power_t(Pxx_ij, mask_ij, fallback_noise_psd)
    
    P_signal_in_bin_t = P_signal_and_noise_in_bin_t - P_noise_in_bin_t
    return np.mean(P_signal_in_bin_t), np.mean(P_noise_in_bin_t)

def set_std_from_snr(waveform : NDArray[np.float64], mask : NDArray[np.uint8], snr_db : float) -> float:
    if not np.any(mask != 0):
        print("Warning no signal in mask")
        return 1
    
    freqs, time, Zxx = scipy_stft_complex_psd(waveform)
    freqs, Zxx = frequency_band(freqs, Zxx)
    
    Pxx = np.abs(Zxx)**2
    global_noise_psd = np.sum(2 * df * Pxx[mask == 0]) / np.sum(2 * df * (mask == 0))
    signal_power, _ = compute_P_moy(Pxx, mask, global_noise_psd)

    M_in = mask != 0
    B_in_t = np.sum(2 * df * M_in, axis = 0)

    valid = B_in_t > 0

    snr = 10 ** (snr_db / 10)
    
    return np.sqrt(signal_power*SAMPLING_RATE/(snr*np.mean(B_in_t[valid])))