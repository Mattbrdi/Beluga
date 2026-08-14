import numpy as np 
import numpy.random as rd
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt

from time_frequency_mask.configuration import DURATION, SAMPLING_RATE

def impulsive_noise_generato_per_channel(duration : float, sampling_rate : int):
    pass

def impulsive_noise_generator_per_audio_array(audio_array : NDArray[np.float64], duration : float, sampling_rate : int, debug = False) -> NDArray[np.float64]:
    pass

def gaussian_noise_generator(std : float, duration : float = DURATION, sampling_rate : int = SAMPLING_RATE) -> NDArray[np.float64]:
    noise = rd.normal(loc = 0, scale = std, size =  int(duration * sampling_rate))
    return noise

def gaussian_noise_generator_2(std : float, duration : float = DURATION, sampling_rate : int = SAMPLING_RATE) -> NDArray[np.float64]:
    noise = rd.normal(loc = 0, scale = std, size =  int(duration * sampling_rate))
    
    noise_2 = rd.normal(loc = 0, scale = 4*std, size =  int(duration * sampling_rate))
    low = rd.uniform()*1000 + 1000
    sos = butter(4, low, btype="low", fs= sampling_rate, output='sos')
    noise_2 = sosfilt(sos, noise_2, axis=0)
    return noise + 2*noise_2
    