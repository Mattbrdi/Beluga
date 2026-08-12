import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt


def bandpass_filter(
    waveform: NDArray[np.float64], sampling_rate: float, low: float, high: float
) -> NDArray[np.float64]:
    sos = butter(4, (low, high), "band", fs=sampling_rate, output="sos")
    return sosfilt(sos, waveform)
