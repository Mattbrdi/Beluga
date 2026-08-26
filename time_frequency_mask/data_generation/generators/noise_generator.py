import numpy as np
import numpy.random as rd
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt


def impulsive_noise_generato_per_channel(duration: float, sampling_rate: int):
    pass


def impulsive_noise_generator_per_audio_array(
    audio_array: NDArray[np.float64], duration: float, sampling_rate: int, debug=False
) -> NDArray[np.float64]:
    pass


def gaussian_noise_generator(
    std: float, duration: float, sampling_rate: int, is_low_band_noise: bool
) -> NDArray[np.float64]:
    noise = rd.normal(loc=0, scale=std, size=int(duration * sampling_rate))

    if is_low_band_noise and rd.random() > 0.25:
        lb_noise = rd.normal(loc=0, scale=4 * std, size=int(duration * sampling_rate))
        low_thr = rd.uniform() * 1000 + 1000

        sos = butter(4, low_thr, btype="low", fs=sampling_rate, output="sos")
        lb_noise = sosfilt(sos, lb_noise, axis=0)
        noise += 2 * lb_noise
    return noise
