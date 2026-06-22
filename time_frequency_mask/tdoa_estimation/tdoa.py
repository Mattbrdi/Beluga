import numpy as np 
from numpy.typing import NDArray

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from time_frequency_mask.configuration import SAMPLING_RATE, N_FFT, MAX_TDOA, DURATION
from scipy.signal import correlate

def windowed_correlation(audio_one : NDArray[np.float64], audio_two : NDArray[np.float64], idx_0 : int, win_size = N_FFT):
    
    if len(audio_one) != len(audio_two):
        raise ValueError(f"Audio length mismatch got {len(audio_one)} and {len(audio_two)}")

    L = len(audio_one)

    n_tdoa = int(np.ceil(SAMPLING_RATE*MAX_TDOA))

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

def compute_tdoa(audio_one : NDArray[np.float64], audio_two : NDArray[np.float64], idx_0 : int, win_size = N_FFT):
    n_tdoa = int(np.ceil(SAMPLING_RATE*MAX_TDOA))
    one_full_two_windowed = windowed_correlation(audio_one, audio_two, idx_0, win_size)
    two_full_one_windowed = windowed_correlation(audio_two, audio_one, idx_0, win_size)

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

def main():
    array_1 = np.arange(-500,500,1).astype(int)

    for i in range(-100,100):
        array_2 = np.arange(-500+i,450,1).astype(int)

        print(i,":" ,compute_tdoa(array_1, array_2, 400, 200))

if __name__ == "__main__":
    main()


    


