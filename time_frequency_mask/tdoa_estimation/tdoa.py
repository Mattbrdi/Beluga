import numpy as np 
from numpy.typing import NDArray

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from time_frequency_mask.configuration import SAMPLING_RATE, N_FFT, MAX_TDOA, DURATION
from time_frequency_mask.tdoa_estimation.blob import Blob
from time_frequency_mask.data_generation.core.preprocess import bandpass_filter
from scipy.signal import correlate

def windowed_correlation(audio_one : NDArray[np.float64], audio_two : NDArray[np.float64], idx_0 : int, win_size = N_FFT, max_tdoa = MAX_TDOA, sampling_rate = SAMPLING_RATE):
    
    if len(audio_one) != len(audio_two):
        raise ValueError(f"Audio length mismatch got {len(audio_one)} and {len(audio_two)}")

    L = len(audio_one)

    n_tdoa = int(np.ceil(sampling_rate*max_tdoa))

    if win_size < 2*n_tdoa +1:
        raise ValueError(f"Incorrect win_size too small to compute accurate tdoa got {win_size} smaller than {2*n_tdoa +1}")

    w_idx_start = int(idx_0 - win_size // 2) 
    w_idx_end = w_idx_start + win_size

    if w_idx_start < n_tdoa or w_idx_end > L - n_tdoa:
        raise ValueError(f"Incorrect w_idx_start or w_idx_end got {w_idx_start} and {w_idx_end}") 

    idx_start = w_idx_start - n_tdoa 
    idx_end = w_idx_end + n_tdoa
    if idx_start < 0 or idx_end > L:
        raise ValueError(f"Incorrect idx_start or idx_end got {idx_start} and {idx_end}")
    
    audio_one_windowed = audio_one[idx_start:idx_end]
    audio_two_windowed = audio_two[w_idx_start:w_idx_end]

    N = len(audio_two_windowed)
    def corr(k):
        return np.dot(audio_one_windowed[k:k+N], audio_two_windowed) / np.sqrt(np.dot(audio_one_windowed[k:k+N], audio_one_windowed[k:k+N]))
 
    norm2_s2_w = np.sqrt(np.dot(audio_two_windowed, audio_two_windowed)) 
    cross_corr = np.array([corr(k) for k in range(2*n_tdoa +1)]).astype(np.float64)
    return cross_corr / norm2_s2_w

def compute_tdoa(audio_one : NDArray[np.float64], audio_two : NDArray[np.float64], idx_0 : int, win_size = N_FFT, max_tdoa = MAX_TDOA, sampling_rate = SAMPLING_RATE):
    n_tdoa = int(np.ceil(sampling_rate*max_tdoa))
    one_full_two_windowed = windowed_correlation(audio_one, audio_two, idx_0, win_size, max_tdoa, sampling_rate)
    two_full_one_windowed = windowed_correlation(audio_two, audio_one, idx_0, win_size, max_tdoa, sampling_rate)

    two_full_one_windowed = np.flipud(two_full_one_windowed)

    cross_corr = one_full_two_windowed + two_full_one_windowed

    tdoa_idx = np.argmax(cross_corr) - n_tdoa
    
    return -tdoa_idx

def compute_cross_corr(audio_one : NDArray[np.float64], audio_two : NDArray[np.float64], idx_0 : int, win_size = N_FFT):
    one_full_two_windowed = windowed_correlation(audio_one, audio_two, idx_0, win_size)
    two_full_one_windowed = windowed_correlation(audio_two, audio_one, idx_0, win_size)

    two_full_one_windowed = np.flipud(two_full_one_windowed)
    cross_corr = one_full_two_windowed + two_full_one_windowed
    # return (cross_corr)/np.std(cross_corr)
    return (cross_corr)

def compute_cross_corr_from_blob(audio_one : NDArray[np.float64], audio_two : NDArray[np.float64], blobs : list[Blob], sampling_rate = SAMPLING_RATE) -> NDArray[np.float64]:
    
    correlations = []
    for i, blob in enumerate(blobs):
        tmin_idx = blob.tmin_idx
        tmax_idx = blob.tmax_idx

        idx_center = (tmin_idx + tmax_idx) // 2
        win_size = tmax_idx - tmin_idx

        fmin = blob.fmin
        fmax = blob.fmax
        audio_one_filtered = bandpass_filter(audio_one, sampling_rate, fmin, fmax)
        audio_two_filtered = bandpass_filter(audio_two, sampling_rate, fmin, fmax)

        correlations.append(blob.area*compute_cross_corr(audio_one_filtered, audio_two_filtered, idx_center, win_size))

    return np.sum(correlations, axis=0)

def main():
    array_1 = np.arange(-500,500,1).astype(int)

    for i in range(-100,100):
        array_2 = np.arange(-500+i,450,1).astype(int)

        print(i,":" ,compute_tdoa(array_1, array_2, 400, 200))

if __name__ == "__main__":
    main()


    


