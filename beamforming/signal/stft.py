from dataclasses import dataclass


import numpy as np
from numpy.typing import NDArray
from scipy.signal import stft


from beamforming.config import *


@dataclass(frozen=True)
class STFTConfig:

    def __post_init__(self) -> None:
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")

        if self.hop_length > self.n_fft:
            raise ValueError("hop_length cannot exceed n_fft")

        if self.min_freq < 0:
            raise ValueError("min_freq must be non-negative")

        if self.max_freq <= self.min_freq:
            raise ValueError("max_freq must be greater than min_freq")

    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    window: str = "hann"

    boundary: str | None = None
    padded: bool = False

    min_freq: float = 500
    max_freq: float = 20000

    @property
    def noverlap(self) -> int:
        return self.n_fft - self.hop_length


def compute_band_stft(
    signal: NDArray[np.float64],
    stft_config: STFTConfig,
    sampling_rate: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]]:
    freqs, times, Zxx = stft(
        signal,
        fs=sampling_rate,
        window=stft_config.window,
        nperseg=stft_config.n_fft,
        noverlap=stft_config.n_fft - stft_config.hop_length,
        nfft=stft_config.n_fft,
        detrend=False,
        return_onesided=True,
        boundary=stft_config.boundary,
        padded=stft_config.padded,
        axis=-1,
    )

    freq_mask = (freqs >= stft_config.min_freq) & (freqs <= stft_config.max_freq)

    return (freqs[freq_mask], times, Zxx[..., freq_mask, :].transpose(1, 0, 2))
