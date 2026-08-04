import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
from scipy.signal import hilbert

from beamforming.configuration import *
from beamforming.classes import *
from time_frequency_mask.data_generation.models.mask import AudioMask
from time_frequency_mask.masknet.run_inference import get_mask_from_array
from time_frequency_mask.tdoa_estimation.blob import (
    Blob,
    output_blobs_from_mask,
    blob_filtering_heuristic,
)
from time_frequency_mask.data_generation.core.preprocess import bandpass_filter
from time_frequency_mask.configuration import MAX_TDOA
from beamforming.mathematics.beamformer import (
    _get_theta_phi_coarse,
    _doa_from_power,
    _get_steering_vector,
    mvdr,
    music,
)


def masked_mvdr(
    tetrahedra: TetrahedralArray, signal: NDArray[np.float64], mask : AudioMask
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """MVDR beamformer for power computation of narrowband signals

    Parameters
    ----------
    fc : int
        central frequency of narrowband signal
    tetrahedra : TetrahedralArray
        Tetrahedra used for beamforming
    signal : NDArray
        4 x N signal containing received signal by the four hydrophones

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray]
        returns power and corresponding angles
    """
    power_dB, Theta, Phi = None, None, None

    blobs = output_blobs_from_mask(mask)
    blobs = blob_filtering_heuristic(blobs)

    N = signal.shape[1]

    Powers = []

    for i, blob in enumerate(blobs):
        tmin_idx = int(max(0, blob.tmin_idx - MAX_TDOA))
        tmax_idx = int(min(N, blob.tmax_idx + MAX_TDOA))

        fmin = blob.fmin
        fmax = blob.fmax

        fc = (fmin + fmax) / 2

        segment = np.copy(signal)
        segment = bandpass_filter(segment, SAMPLING_RATE, fmin, fmax)

        power_dB, Theta, Phi = mvdr(fc, tetrahedra, segment[:, tmin_idx:tmax_idx])
        Powers.append(power_dB)

    power_dB = np.stack(Powers, axis=0).mean(axis=0)
    return power_dB, Theta, Phi


def masked_mvdr_doa(
    tetrahedra: TetrahedralArray, signal: NDArray[np.float64], mask : AudioMask
) -> NDArray[np.float64]:
    """MVDR Beamformer for DOA finding of narrowband signals

    Parameters
    ----------
    fc : int
        central frequency of narrowband signal
    tetrahedra : TetrahedralArray
        Tetrahedra used for beamforming
    signal : NDArray
        4 x N signal containing received signal by the four hydrophones

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray]
        returns DOA
    """
    return _doa_from_power(*masked_mvdr(tetrahedra, signal, mask))


def masked_music(
    tetrahedra: TetrahedralArray, signal: NDArray[np.float64], mask : AudioMask
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """MVDR beamformer for power computation of narrowband signals

    Parameters
    ----------
    fc : int
        central frequency of narrowband signal
    tetrahedra : TetrahedralArray
        Tetrahedra used for beamforming
    signal : NDArray
        4 x N signal containing received signal by the four hydrophones

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray]
        returns power and corresponding angles
    """
    power_dB, Theta, Phi = None, None, None

    blobs = output_blobs_from_mask(mask)
    blobs = blob_filtering_heuristic(blobs)

    N = signal.shape[1]

    Powers = []

    for i, blob in enumerate(blobs):
        tmin_idx = int(max(0, blob.tmin_idx - MAX_TDOA))
        tmax_idx = int(min(N, blob.tmax_idx + MAX_TDOA))

        fmin = blob.fmin
        fmax = blob.fmax

        fc = (fmin + fmax) / 2

        segment = np.copy(signal)
        segment = bandpass_filter(segment, SAMPLING_RATE, fmin, fmax)

        power_dB, Theta, Phi = music(fc, tetrahedra, segment[:, tmin_idx:tmax_idx])
        Powers.append(power_dB)

    power_dB = np.stack(Powers, axis=0).mean(axis=0)
    return power_dB, Theta, Phi


def masked_music_doa(
    tetrahedra: TetrahedralArray, signal: NDArray[np.float64], mask : AudioMask
) -> NDArray[np.float64]:
    """MVDR Beamformer for DOA finding of narrowband signals

    Parameters
    ----------
    fc : int
        central frequency of narrowband signal
    tetrahedra : TetrahedralArray
        Tetrahedra used for beamforming
    signal : NDArray
        4 x N signal containing received signal by the four hydrophones

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray]
        returns DOA
    """
    return _doa_from_power(*masked_music(tetrahedra, signal, mask))
