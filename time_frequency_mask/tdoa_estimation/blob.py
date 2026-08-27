import numpy as np 
from numpy.typing import NDArray
import cv2 as cv

from time_frequency_mask.data_generation.models.mask import AudioMask
from time_frequency_mask.plotter import plot_mask
 
from time_frequency_mask.config import Parameters

class Blob():
    def __init__(self, data : NDArray[np.uint8], stats, parameters : Parameters, area = None):
        audio, array, stft = parameters.audio, parameters.array, parameters.stft

        self.data = data
        n_freqs, n_times = data.shape
        if area is None:
            self.area = n_freqs * n_times
        else:
            self.area = area
        self.stats = stats
        if data is None:
            if audio.duration is None:
                raise ValueError(f"Cannot create default blob if no default duration is provided")
            self.fmin = audio.min_freq
            self.fmax = audio.max_freq
            self.tmin = 0
            self.tmax = audio.duration
            self.tmin_idx = 0
            self.tmax_idx = audio.num_samples

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

            duration = ((stft.hop_length*(n_times - 1)) + stft.n_fft) / audio.sampling_rate  
            start_freq_idx = stft.freq_index(audio.sampling_rate, audio.min_freq)

            h = min(h, stft.n_fft - start_freq_idx)
            y = y + start_freq_idx
            self.fmin = audio.sampling_rate / stft.n_fft * y
            self.fmax = audio.sampling_rate / stft.n_fft * (y + h)

            self.tmin = float(duration / n_times * x) 
            self.tmax = float(duration / n_times * (x + w))
            if x == 0:
                self.tmin += array.max_tdoa

            if x + w == n_times:
                self.tmax -=array.max_tdoa

            #TODO: Vérifier que ça, ça marche bien:
            self.tmin_idx = max(int(np.ceil(self.tmin * audio.sampling_rate)),0)
            self.tmax_idx = min(int(np.floor(self.tmax * audio.sampling_rate)), int(audio.sampling_rate*(duration - array.max_tdoa)))

            # print(self.fmin, self.fmax)
            # print(self.tmin, self.tmax)


def output_blobs_from_mask(mask : NDArray[np.bool_], parameters : Parameters, area_thr=30) -> list[Blob]:
    mask_data = mask.astype(np.uint8)

    masks = []
    N, labels, stats, centroids = cv.connectedComponentsWithStats(mask_data)
    
    mean_area = np.mean([stats[i, cv.CC_STAT_AREA] for i in range(1, N)])

    for i in range(1,N):
        area = stats[i, cv.CC_STAT_AREA]

        if area < area_thr or area < 0.1*mean_area:
            continue
    
        label = (np.array(labels) == i).astype(np.uint8)
        # plot_mask(label)
        masks.append(Blob(label, stats, parameters, stats[i, cv.CC_STAT_AREA]))

    return masks

def output_mask_from_blobs(blobs : list[Blob], n_freqs : int, n_times : int) -> NDArray[np.bool_]:
    mask = np.zeros((n_freqs, n_times), dtype=np.bool_)

    for blob in blobs:
        mask = mask | (blob.data != 0)

    return mask

def blob_filtering_heuristic(
    blobs : list[Blob],
    min_freq : float,
    max_blobs_count : int = 7,
    min_area : int = 7*7,
    min_total_area : int = 12*12,
) -> list[Blob]:
    output_blobs = []
    mean_area = np.mean([blob.area for blob in blobs])
    count = len(blobs)

    argsort = np.flip(np.argsort([blob.area for blob in blobs]))
    sorted_blobs = [blobs[argsort[idx]] for idx in range(min(count, max_blobs_count))]

    for blob in sorted_blobs:
        freq_cond = blob.fmin < 1.05* min_freq

        min_area_cond = blob.area < min_area
        
        mean_area_cond = blob.area < 0.2*mean_area

        if not freq_cond and not min_area_cond and not mean_area_cond:
            output_blobs.append(blob)

    if np.sum([blob.area for blob in output_blobs]) < min_total_area:
        raise ValueError(f"Error total area of mask is too small for masked based tdoa got {np.sum([blob.area for blob in output_blobs])} smaller than min_total_area: {min_total_area}")

    if len(output_blobs) == 0:
        raise ValueError(f"Error output_blobs is empty unable to perform masked TDOA on current sample")
    return output_blobs


