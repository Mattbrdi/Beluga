import numpy as np 
from numpy.typing import NDArray
import cv2 as cv

from time_frequency_mask.data_generation.models.mask import AudioMask
from time_frequency_mask.plotter import plot_mask
 
from time_frequency_mask.configuration import MIN_FREQ, MAX_TDOA, MAX_FREQ, SAMPLING_RATE, DURATION, N_FFT, DURATION, N_TIMES, START_FREQ_IDX, N_FREQS

class Blob():
    def __init__(self, data : NDArray[np.uint8], stats, area = N_FREQS*N_TIMES):
        self.data = data
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
            if x < 0 or x >= N_TIMES:
                raise ValueError(f"Error incorrect x start boundingRect index not between 0 and N_TIMES {N_TIMES} got {x}")
            
            if y < 0 or y >= N_FREQS:
                raise ValueError(f"Error incorrect y start boundingRect index not between 0 and N_TIMES {N_FREQS} got {y}")

            if x + w <= 0 or x + w > N_TIMES:
                raise ValueError(f"Error incorrect w boundingRect index not between 0 and N_TIMES {N_TIMES} got {x+w}")
            
            if y + h <=0 or  y + h > N_FREQS:
                raise ValueError(f"Error incorrect h boundingRect index not between 0 and N_TIMES {N_FREQS} got {x+h}")



            h = min(h, N_FFT - START_FREQ_IDX)
            y = y + START_FREQ_IDX
            self.fmin = SAMPLING_RATE / N_FFT * y
            self.fmax = SAMPLING_RATE / N_FFT * (y + h)

            self.tmin = float(DURATION / N_TIMES * x) 
            self.tmax = float(DURATION / N_TIMES * (x + w))
            if x == 0:
                self.tmin += MAX_TDOA

            if x + w == N_TIMES:
                self.tmax -=MAX_TDOA

            #TODO: Vérifier que ça, ça marche bien:
            self.tmin_idx = max(int(np.ceil(self.tmin * SAMPLING_RATE)),0)
            self.tmax_idx = min(int(np.floor(self.tmax * SAMPLING_RATE)), int(SAMPLING_RATE*(DURATION - MAX_TDOA)))

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

def blob_filtering_heuristic(blobs : list[Blob], max_blobs_count = 7, min_area = 10*10) -> list[Blob]:
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

    return output_blobs


