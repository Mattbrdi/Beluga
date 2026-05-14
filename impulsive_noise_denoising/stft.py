import numpy as np

from scipy.signal import stft

def scipy_stft(canal, frame_rate, n_fft=4096, hop_length=2048):
    """Compute an STFT using scipy. Returns magnitude shaped as frequency x time."""
    if len(canal) < n_fft:
        raise ValueError(f"Audio slice is too short for n_fft={n_fft}. Got {len(canal)} samples.")
    noverlap = n_fft - hop_length

    freqs, times, Zxx = stft(
        canal,
        fs=frame_rate,
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

def scipy_db_spectrogram(canal, frame_rate, n_fft=4096, hop_length=2048, gain_db=0):
    freqs, times, S = scipy_stft(canal, frame_rate, n_fft=n_fft, hop_length=hop_length)
    D = 20 * np.log10(np.maximum(2 * S, 1e-12)) + gain_db
    return freqs, times, D

def scipy_spectrogram(canal, frame_rate, n_fft=4096, hop_length=2048, gain_db=0):
    freqs, times, S = scipy_stft(canal, frame_rate, n_fft=n_fft, hop_length=hop_length)
    return freqs, times, 2 * S

def frequency_band(freqs, D, fmin=2500, fmax=12500):
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(freq_mask):
        raise ValueError(f"No frequency bins found between {fmin} Hz and {fmax} Hz")
    return freqs[freq_mask], D[freq_mask, :]