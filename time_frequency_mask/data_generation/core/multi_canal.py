import numpy as np
from numpy.typing import NDArray

from time_frequency_mask.configuration import MAX_TDOA, SAMPLING_RATE, HOP_LENGTH, M

def set_tdoa(waveform : NDArray[np.float64], mask : NDArray[np.uint8], shift : float = 0, sampling_rate = SAMPLING_RATE) -> tuple[NDArray[np.float64], NDArray[np.uint8]]:
    if abs(shift) > MAX_TDOA:
        raise ValueError(f"Got incorrect shift value, got {shift} but needs to be between +-{MAX_TDOA}")
    
    if shift == 0:
        return waveform, mask
    
    shift_index = int(round(shift * sampling_rate))

    time_bin_shift = int(round(shift_index / HOP_LENGTH))

    shifted_waveform = np.zeros_like(waveform, dtype=np.float64)
    shifted_mask = np.zeros_like(mask, dtype=np.uint8)

    def shift_bounds(length, shift_index):
        src_start = max(0, -shift_index)
        dst_start = max(0, shift_index)

        src_end = length - dst_start
        dst_end = length - src_start
    
        return src_start, src_end, dst_start, dst_end

    src_start, src_end, dst_start, dst_end = shift_bounds(len(waveform), shift_index)
    shifted_waveform[dst_start:dst_end] = waveform[src_start: src_end]

    src_start, src_end, dst_start, dst_end = shift_bounds(mask.shape[1], time_bin_shift)
    shifted_mask[:, dst_start:dst_end] = mask[:, src_start: src_end]

    return shifted_waveform, shifted_mask    

def set_channels_tdoas(waveforms : list[NDArray[np.float64]], masks : list[NDArray[np.uint8]], shifts : list[float], num_microphones = M) -> tuple[list[NDArray[np.float64]], list[NDArray[np.uint8]]]:
    """Set shifts by taking hydrophone 1 as reference and shifting hydrophone 2, 3 and 4.

    Parameters
    ----------
    waveforms : list[NDArray[np.float64]]
        waveforms of the hydrophone 
    masks : list[NDArray[np.uint8]]
        masks if needing to be changed
    shifts : list[float]
        shift values
    """
    if len(shifts) !=(num_microphones - 1):
        raise ValueError(f"Need 3 shifts value but got {len(shifts)} ")

    for shift in shifts:
        if abs(shift) > MAX_TDOA:
            raise ValueError(f"Got incorrect shift value, got {shift} but need to be between +-{MAX_TDOA}")
        
    if all(x == 0 for x in shifts):
        return waveforms, masks
    
    shifted_waveforms = [waveforms[0]]
    shifted_masks = [masks[0]]

    for waveform, mask, shift in zip(waveforms[1:], masks[1:], shifts):
        shifted_waveform, shifted_mask = set_tdoa(waveform, mask, shift)
        
        shifted_waveforms.append(shifted_waveform)
        shifted_masks.append(shifted_mask)

    return shifted_waveforms, shifted_masks
