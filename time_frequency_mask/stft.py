import numpy as np
from numpy.typing import NDArray

from librosa import stft, frames_to_time, fft_frequencies
import scipy.signal as signal

from time_frequency_mask.configuration import SAMPLING_RATE, N_FFT, HOP_LENGTH, WINDOW_LIBROSA, MIN_FREQ, MAX_FREQ

def frequency_band(freqs : NDArray[np.float64], D : NDArray[np.complex128], fmin : float = MIN_FREQ, fmax : float = MAX_FREQ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    freq_mask = (freqs >= fmin) & (freqs <= fmax) 
    if not np.any(freq_mask):
        raise ValueError(f"No frequency bins found between {fmin} Hz and {fmax} Hz")
    return freqs[freq_mask], D[freq_mask, :]

def librosa_stft(waveform : NDArray[np.float64], n_fft=N_FFT, hop_length=HOP_LENGTH, window=WINDOW_LIBROSA) -> tuple[NDArray[np.float64,], NDArray[np.float64], NDArray[np.complex128]]:
    D = stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window, center=False)

    times = frames_to_time(np.arange(D.shape[1]), sr=SAMPLING_RATE, hop_length=HOP_LENGTH)
    freqs = fft_frequencies(sr=SAMPLING_RATE, n_fft=N_FFT)

    return freqs, times, D

def compute_power_non_zero(D : NDArray[np.complex128]) -> float:
    non_zero = D != 0
    n_bins = np.count_nonzero(non_zero)

    if n_bins == 0:
        return 0.0
    
    energy = np.abs(D[non_zero]) ** 2
    window = signal.get_window("hann", N_FFT, fftbins=True)

    # return float(np.sum(energy) / n_bins)
    return float(np.sum(energy) / (np.sum(window) **2))

def get_mask_stft(D : NDArray[np.complex128], mask : NDArray[np.uint8]) -> NDArray[np.complex128]:
    if D.shape != mask.shape:
        raise ValueError(f"Incorrect mask format got {mask.shape} for mask instead of {D.shape}")

    return D * (mask > 0)

def compute_power_from_waveform_and_mask(waveform : NDArray[np.float64], mask : NDArray[np.uint8], n_fft : float = N_FFT, hop_length : float = HOP_LENGTH, window : float = WINDOW_LIBROSA) -> float:
    freqs, times, D = librosa_stft(waveform, n_fft, hop_length, window)

    _, D_band = frequency_band(freqs, D)
    D_masked = get_mask_stft(D_band, mask)

    return compute_power_non_zero(D_masked)

def scipy_stft_complex(canal, frame_rate = SAMPLING_RATE, n_fft=N_FFT, hop_length = HOP_LENGTH) -> tuple[NDArray[np.float64,], NDArray[np.float64], NDArray[np.complex128]]:
    """Compute an STFT using scipy. Returns complex valued stft shaped as frequency x time."""
    if len(canal) < n_fft:
        raise ValueError(f"Audio slice is too short for n_fft={n_fft}. Got {len(canal)} samples.")
    noverlap = n_fft - hop_length

    freqs, times, Zxx = signal.stft(
        canal,
        fs= frame_rate,
        window='hann',
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        detrend=False,
        return_onesided=True,
        boundary=None,
        padded=False,
    )

    return freqs, times, Zxx

def scipy_stft(canal, frame_rate = SAMPLING_RATE, n_fft=N_FFT, hop_length = HOP_LENGTH) -> tuple[NDArray[np.float64,], NDArray[np.float64], NDArray[np.float64]]:
    """Compute an STFT using scipy. Returns magnitude shaped as frequency x time."""
    if len(canal) < n_fft:
        raise ValueError(f"Audio slice is too short for n_fft={n_fft}. Got {len(canal)} samples.")
    noverlap = n_fft - hop_length

    freqs, times, Zxx = signal.stft(
        canal,
        fs= frame_rate,
        window='hann',
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        detrend=False,
        return_onesided=True,
        boundary=None,
        padded=False,
    )

    S = np.abs(Zxx)
    return freqs, times, S

def scipy_db_spectrogram(canal, frame_rate=SAMPLING_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, gain_db=0) -> tuple[NDArray[np.float64,], NDArray[np.float64], NDArray[np.float64]]:
    freqs, times, S = scipy_stft(canal, frame_rate, n_fft=n_fft, hop_length=hop_length)
    D = 20 * np.log10(np.maximum(2 * S, 1e-12)) + gain_db
    return freqs, times, D

def scipy_spectrogram(canal, frame_rate=SAMPLING_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH) -> tuple[NDArray[np.float64,], NDArray[np.float64], NDArray[np.float64]]:
    freqs, times, S = scipy_stft(canal, frame_rate, n_fft=n_fft, hop_length=hop_length)
    return freqs, times, 2 * S