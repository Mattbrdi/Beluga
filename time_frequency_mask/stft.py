import numpy as np
from numpy.typing import NDArray


import scipy.signal as signal

from time_frequency_mask.config import STFTParamters

def frequency_band(freqs : NDArray[np.float64], D : NDArray, fmin : float, fmax : float) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    freq_mask = (freqs >= fmin) & (freqs <= fmax) 
    if not np.any(freq_mask):
        raise ValueError(f"No frequency bins found between {fmin} Hz and {fmax} Hz")
    return freqs[freq_mask], D[freq_mask, :]

def librosa_stft(canal : NDArray[np.float64], sampling_rate : float, stft_parameters : STFTParamters) -> tuple[NDArray[np.float64,], NDArray[np.float64], NDArray[np.complex128]]:
    from librosa import stft, frames_to_time, fft_frequencies
    D = stft(canal, n_fft=stft_parameters.n_fft, hop_length=stft_parameters.hop_length, window=stft_parameters.window, center=False)

    times = frames_to_time(np.arange(D.shape[1]), sr=sampling_rate, hop_length=stft_parameters.hop_length)
    freqs = fft_frequencies(sr=sampling_rate, n_fft=stft_parameters.n_fft)

    return freqs, times, D

def get_mask_stft(D : NDArray[np.complex128], mask : NDArray[np.uint8]) -> NDArray[np.complex128]:
    if D.shape != mask.shape:
        raise ValueError(f"Incorrect mask format got {mask.shape} for mask instead of {D.shape}")

    return D * (mask > 0)

def scipy_stft_complex(canal, sampling_rate, stft_parameters : STFTParamters) -> tuple[NDArray[np.float64,], NDArray[np.float64], NDArray[np.complex128]]:
    """Compute an STFT using scipy. Returns complex valued stft shaped as frequency x time."""
    if len(canal) < stft_parameters.n_fft:
        raise ValueError(f"Audio slice is too short for n_fft={stft_parameters.n_fft}. Got {len(canal)} samples.")
    noverlap = stft_parameters.n_fft - stft_parameters.hop_length

    freqs, times, Zxx = signal.stft(
        canal,
        fs= sampling_rate,
        window=stft_parameters.window,
        nperseg=stft_parameters.n_fft,
        noverlap=noverlap,
        nfft=stft_parameters.n_fft,
        detrend=stft_parameters.detrend,
        return_onesided=True,
        boundary=stft_parameters.boundary,
        padded=stft_parameters.padded,
    )

    return freqs, times, Zxx

def scipy_stft_complex_psd(canal, frame_rate, stft_parameters : STFTParamters) -> tuple[NDArray[np.float64,], NDArray[np.float64], NDArray[np.complex128]]:
    """Compute an STFT using scipy. Returns complex valued stft shaped as frequency x time."""
    if canal.shape[-1] < stft_parameters.n_fft:
        raise ValueError(f"Audio slice is too short for n_fft={stft_parameters.n_fft}. Got {canal.shape[-1]} samples.")
    noverlap = stft_parameters.n_fft - stft_parameters.hop_length

    freqs, times, Zxx = signal.stft(
        canal,
        fs= frame_rate,
        window=stft_parameters.window,
        nperseg=stft_parameters.n_fft,
        noverlap=noverlap,
        nfft=stft_parameters.n_fft,
        detrend=stft_parameters.detrend,
        return_onesided=True,
        boundary=stft_parameters.boundary,
        padded=stft_parameters.padded,
        scaling="psd"
    )    

    return freqs, times, Zxx

def scipy_stft(canal, sampling_rate, stft_parameters : STFTParamters) -> tuple[NDArray[np.float64,], NDArray[np.float64], NDArray[np.float64]]:
    """Compute an STFT using scipy. Returns magnitude shaped as frequency x time."""
    if len(canal) < stft_parameters.n_fft:
        raise ValueError(f"Audio slice is too short for n_fft={stft_parameters.n_fft}. Got {len(canal)} samples.")
    noverlap = stft_parameters.n_fft - stft_parameters.hop_length

    freqs, times, Zxx = signal.stft(
        canal,
        fs= sampling_rate,
        window=stft_parameters.window,
        nperseg=stft_parameters.n_fft,
        noverlap=noverlap,
        nfft=stft_parameters.n_fft,
        detrend=stft_parameters.detrend,
        return_onesided=True,
        boundary=stft_parameters.boundary,
        padded=stft_parameters.padded,
    )

    S = np.abs(Zxx)
    return freqs, times, S

def scipy_db_spectrogram(canal, frame_rate, stft_parameters : STFTParamters, gain_db=0) -> tuple[NDArray[np.float64,], NDArray[np.float64], NDArray[np.float64]]:
    freqs, times, S = scipy_stft(canal, frame_rate, stft_parameters)
    D = 20 * np.log10(np.maximum(2 * S, 1e-12)) + gain_db
    return freqs, times, D

def scipy_spectrogram(canal, frame_rate, stft_parameters : STFTParamters) -> tuple[NDArray[np.float64,], NDArray[np.float64], NDArray[np.float64]]:
    freqs, times, S = scipy_stft(canal, frame_rate, stft_parameters)
    return freqs, times, 2 * S