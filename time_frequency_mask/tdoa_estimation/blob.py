import numpy as np 
from numpy.typing import NDArray
import cv2 as cv

from time_frequency_mask.data_generation.models.mask import AudioMask
from time_frequency_mask.plotter import plot_mask
 
from time_frequency_mask.configuration import MIN_FREQ, MAX_TDOA, MAX_FREQ, SAMPLING_RATE, DURATION, N_FFT, DURATION, N_TIMES, START_FREQ_IDX, N_FREQS, HOP_LENGTH

class Blob():
    def __init__(self, data : NDArray[np.uint8], stats, area = None, max_tdoa : float = MAX_TDOA, sampling_rate : float = SAMPLING_RATE):
        self.data = data
        n_freqs, n_times = data.shape
        if area is None:
            self.area = n_freqs * n_times
        else:
            self.area = area
        self.stats = stats
        if data is None:
            self.fmin = MIN_FREQ
            self.fmax = MAX_FREQ
            self.tmin = 0
            self.tmax = DURATION
            self.tmin_idx = 0
            self.tmax_idx = int(DURATION * SAMPLING_RATE)

        else:
            x, y, w, h = cv.boundingRect(self.data)
            # print(x,y,w,h)
            if x < 0 or x >= n_times:
                raise ValueError(f"Error incorrect x start boundingRect index not between 0 and n_times {n_times} got {x}")
            
            if y < 0 or y >= n_freqs:
                raise ValueError(f"Error incorrect y start boundingRect index not between 0 and n_freqs {n_freqs} got {y}")

            if x + w <= 0 or x + w > n_times:
                raise ValueError(f"Error incorrect w boundingRect index not between 0 and n_times {n_times} got {x+w}")
            
            if y + h <=0 or  y + h > n_freqs:
                raise ValueError(f"Error incorrect h boundingRect index not between 0 and n_freqs {n_freqs} got {x+h}")

            duration = ((HOP_LENGTH*(n_times - 1)) + N_FFT) / sampling_rate  

            h = min(h, N_FFT - START_FREQ_IDX)
            y = y + START_FREQ_IDX
            self.fmin = sampling_rate / N_FFT * y
            self.fmax = sampling_rate / N_FFT * (y + h)

            self.tmin = float(duration / n_times * x) 
            self.tmax = float(duration / n_times * (x + w))
            if x == 0:
                self.tmin += max_tdoa

            if x + w == n_times:
                self.tmax -=max_tdoa

            #TODO: Vérifier que ça, ça marche bien:
            self.tmin_idx = max(int(np.ceil(self.tmin * sampling_rate)),0)
            self.tmax_idx = min(int(np.floor(self.tmax * sampling_rate)), int(sampling_rate*(duration - max_tdoa)))

            # print(self.fmin, self.fmax)
            # print(self.tmin, self.tmax)


def output_blobs_from_mask(mask : AudioMask, area_thr=30) -> list[Blob]:
    mask_data = mask.data.astype(np.uint8)

    masks = []
    N, labels, stats, centroids = cv.connectedComponentsWithStats(mask_data)
    
    mean_area = np.mean([stats[i, cv.CC_STAT_AREA] for i in range(1, N)])

    for i in range(1,N):
        area = stats[i, cv.CC_STAT_AREA]

        if area < area_thr or area < 0.1*mean_area:
            continue
    
        label = (np.array(labels) == i).astype(np.uint8)
        # plot_mask(label)
        masks.append(Blob(label, stats, stats[i, cv.CC_STAT_AREA]))

    return masks

def output_mask_from_blobs(blobs : list[Blob], n_freqs = N_FREQS, n_times = N_TIMES, sampling_rate = SAMPLING_RATE) -> AudioMask:
    mask = AudioMask.create_empty_mask(n_freqs, n_times, sampling_rate)

    for blob in blobs:
        mask.data = mask.data | (blob.data != 0)

    return mask

def blob_filtering_heuristic(blobs : list[Blob], max_blobs_count = 7, min_area = 7*7, min_total_area = 12*12) -> list[Blob]:
    output_blobs = []
    mean_area = np.mean([blob.area for blob in blobs])
    count = len(blobs)

    argsort = np.flip(np.argsort([blob.area for blob in blobs]))
    sorted_blobs = [blobs[argsort[idx]] for idx in range(min(count, max_blobs_count))]

    for blob in sorted_blobs:
        freq_cond = blob.fmin < 1.05* MIN_FREQ

        min_area_cond = blob.area < min_area
        
        mean_area_cond = blob.area < 0.2*mean_area

        if not freq_cond and not min_area_cond and not mean_area_cond:
            output_blobs.append(blob)

    if np.sum([blob.area for blob in output_blobs]) < min_total_area:
        raise ValueError(f"Error total area of mask is too small for masked based tdoa got {np.sum([blob.area for blob in output_blobs])} smaller than min_total_area: {min_total_area}")

    if len(output_blobs) == 0:
        raise ValueError(f"Error output_blobs is empty unable to perform masked TDOA on current sample")
    return output_blobs


