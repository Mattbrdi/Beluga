import numpy as np
from numpy.typing import NDArray

from scipy.signal import stft
from scipy.signal import hilbert
from beamforming.configuration import *
from beamforming.classes import *

from beamforming.mathematics.beamformer import (
    _get_steering_vector,
    _get_theta_phi_coarse,
    _get_theta_phi_fine,
    _doa_from_power,
    estimate_num_sources,
    _get_steering_vector_per_freq,
)

from time_frequency_mask.configuration import N_FFT, HOP_LENGTH, MIN_FREQ, MAX_FREQ


def wideband_issm_mvdr(
    tetrahedra: TetrahedralArray, signal: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Wideband MVDR Beamformer power computation for wideband signals

    Parameters
    ----------
    tetrahedra : TetrahedralArray
        Tetrahedra used for beamforming
    signal : NDArray
        4 x N signal containing received signal by the four hydrophones

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray]
        returns power and corresponding angles
    """

    noverlap = N_FFT - HOP_LENGTH
    freqs, times, Zxx = stft(
        signal,
        fs=SAMPLING_RATE,
        window="hann",
        nperseg=N_FFT,
        noverlap=noverlap,
        nfft=N_FFT,
        detrend=False,
        return_onesided=True,
        boundary=None,
        padded=False,
        axis=-1,
    )

    freq_mask = (freqs >= MIN_FREQ) & (freqs <= MAX_FREQ)
    freqs, Zxx = freqs[freq_mask], Zxx[:, freq_mask, :]

    Theta, Phi = _get_theta_phi_coarse()
    T, P = Theta.shape

    power_tot = None
    for i, freq in enumerate(freqs):
        s = _get_steering_vector(freq, tetrahedra, Theta, Phi)
        s_flat = s.reshape(4, T * P)  # (4, T*P)

        X = Zxx[:, i]
        R = X @ X.conj().T / X.shape[1]
        loading = 1e-2 * np.trace(R).real / R.shape[0]
        R_loaded = R + loading * np.eye(R.shape[0])

        Rinv = np.linalg.inv(R_loaded)  # (4, T, P)
        Rinv_s = Rinv @ s_flat  # (4, T*P)

        denominator = np.sum(
            s_flat.conj() * Rinv_s,
            axis=0,
            keepdims=True,
        )  # (1, T*P)

        denominator = np.maximum(np.real(denominator), 1e-12)
        power = (1.0 / denominator).reshape(T, P)

        if power_tot is None:
            power_tot = power.copy()
        else:
            power_tot += power

    power_dB = 10 * np.log10(power_tot)
    power_dB -= np.max(power_dB)

    return power_dB, Theta, Phi


def wideband_issm_mvdr_doa(
    tetrahedra: TetrahedralArray, signal: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Wideband MVDR Beamformer for DOA finding of wideband signals

    Parameters
    ----------
    tetrahedra : TetrahedralArray
        Tetrahedra used for beamforming
    signal : NDArray
        4 x N signal containing received signal by the four hydrophones

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    return _doa_from_power(*wideband_issm_mvdr(tetrahedra, signal))


def wideband_issm_music(
    tetrahedra: TetrahedralArray, signal: NDArray[np.float64], num_expected_sources=1
):
    """Wideband MUSIC Beamformer power computation for wideband signals

    Parameters
    ----------
    tetrahedra : TetrahedralArray
        Tetrahedra used for beamforming
    signal : NDArray
        4 x N signal containing received signal by the four hydrophones
    num_expected_sources : float
        number of expected sources in signal

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray]
        returns power and corresponding angles
    """

    noverlap = N_FFT - HOP_LENGTH
    freqs, times, Zxx = stft(
        signal,
        fs=SAMPLING_RATE,
        window="hann",
        nperseg=N_FFT,
        noverlap=noverlap,
        nfft=N_FFT,
        detrend=False,
        return_onesided=True,
        boundary=None,
        padded=False,
        axis=-1,
    )

    freq_mask = (freqs >= MIN_FREQ) & (freqs <= MAX_FREQ)
    freqs, Zxx = freqs[freq_mask], Zxx[:, freq_mask, :]

    Theta, Phi = _get_theta_phi_coarse()
    T, P = Theta.shape

    metric_tot = None
    for i, freq in enumerate(freqs):
        s = _get_steering_vector(freq, tetrahedra, Theta, Phi)
        s_flat = s.reshape(4, T * P)  # (4, T*P)

        X = Zxx[:, i]
        R = X @ X.conj().T / X.shape[1]

        w, v = np.linalg.eig(R)
        eig_val_order = np.argsort(np.abs(w))
        v = v[:, eig_val_order]
        V = np.zeros((4, 4 - num_expected_sources), dtype=np.complex128)
        for i in range(4 - num_expected_sources):
            V[:, i] = v[:, i]
        vvh_s = V @ V.conj().T @ s_flat

        denominator = np.sum(
            s_flat.conj() * vvh_s,
            axis=0,
            keepdims=True,
        )

        metric_flat = 1 / np.maximum(np.abs(denominator), 1e-12)
        metric = metric_flat.reshape(T, P)

        if metric_tot is None:
            metric_tot = metric.copy()
        else:
            metric_tot += metric

    metric_tot = 10 * np.log10(metric_tot)
    metric_tot -= np.max(metric_tot)

    return metric_tot, Theta, Phi


def wideband_issm_music_doa(
    tetrahedra, signal, num_expected_sources=1
) -> NDArray[np.float64]:
    """Wideband MUSIC Beamformer for DOA finding of wideband signals

    Parameters
    ----------
    tetrahedra : TetrahedralArray
        Tetrahedra used for beamforming
    signal : NDArray
        4 x N signal containing received signal by the four hydrophones
    num_expected_sources : int
        number of expected sources in signal

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    return _doa_from_power(
        *wideband_issm_music(
            tetrahedra, signal, num_expected_sources=num_expected_sources
        )
    )


def _compute_cssm_correlation_matrix(
    tetrahedra: TetrahedralArray,
    signal: NDArray[np.float64],
    center_freq: float,
    initial_doa: NDArray[np.float64],
    num_expected_sources: int,
) -> NDArray[np.complex128]:
    """Compute the correlation matrix for CSSM wideband beamformers

    Parameters
    ----------
    tetrahedra : TetrahedralArray
        Tetrahedra used for beamforming
    signal : NDArray
        4 x N signal containing received signal by the four hydrophones
    center_freq : float
        Frequency used to steer the selected frequency bins
    initial_doa : NDArray
        3x1 vector representing the direction of arrival
    num_expected_sources : int
        number of expected sources in signal

    Returns
    -------
    NDArray
        Correlation matrix used for adaptative beamforming
    """

    if initial_doa is None:
        initial_doa = wideband_issm_mvdr_doa(tetrahedra, signal)
    initial_doa = initial_doa / np.linalg.norm(initial_doa)

    initial_theta = np.arccos(np.clip(initial_doa[2], -1.0, 1.0))
    initial_phi = np.mod(
        np.arctan2(initial_doa[1], initial_doa[0]),
        2 * np.pi,
    )

    noverlap = N_FFT - HOP_LENGTH
    freqs, times, Zxx = stft(
        signal,
        fs=SAMPLING_RATE,
        window="hann",
        nperseg=N_FFT,
        noverlap=noverlap,
        nfft=N_FFT,
        detrend=False,
        return_onesided=True,
        boundary=None,
        padded=False,
        axis=-1,
    )

    freq_mask = (freqs >= MIN_FREQ) & (freqs <= MAX_FREQ)
    freqs, Zxx = freqs[freq_mask], Zxx[:, freq_mask, :]
    Zxx = np.swapaxes(Zxx, 0, 1)  # (F, 4, T)

    s_0 = _get_steering_vector(center_freq, tetrahedra, initial_theta, initial_phi)  # 4
    s_ks = _get_steering_vector_per_freq(
        freqs, tetrahedra, initial_theta, initial_phi
    )  # (F, 4)

    R_ks = np.einsum("fmt,fnt->fmn", Zxx, Zxx.conj())  # (F, 4, 4)
    focus_matrixes = s_0 / s_ks  # (F,4)

    R_focused = (
        focus_matrixes[:, :, None]  # (F, 4, 1)
        * R_ks  # (F, 4, 4)
        * focus_matrixes.conj()[:, None, :]  # (F, 1, 4)
    )

    return np.sum(R_focused, axis=0)  # (4, 4)


def _get_power_mvdr(
    steering_vector: NDArray[np.complex128], correlation_matrix: NDArray[np.complex128]
) -> NDArray[np.float64]:
    """MVDR power computation from steering vector and correlation matrix

    Parameters
    ----------
    steering_vector : NDArray[np.complex128]
        Steering vector 4, T, P sized matrix
    correlation_matrix : NDArray[np.complex128]
        correlation matrix 4, 4 matrix used for MVDR algorithm

    Returns
    -------
    NDArray[np.float64]
        Power in db at each scanned angle
    """
    _, T, P = np.shape(steering_vector)
    s_flat = steering_vector.reshape(4, T * P)  # (4, T*P)

    loading = 1e-2 * np.trace(correlation_matrix).real / correlation_matrix.shape[0]
    R_loaded = correlation_matrix + loading * np.eye(correlation_matrix.shape[0])

    Rinv = np.linalg.inv(R_loaded)  # (4, T, P)
    Rinv_s = Rinv @ s_flat  # (4, T*P)

    denominator = np.sum(
        s_flat.conj() * Rinv_s,
        axis=0,
        keepdims=True,
    )  # (1, T*P)

    denominator = np.maximum(np.real(denominator), 1e-12)

    power = (1.0 / denominator).reshape(T, P)
    power_dB = 10 * np.log10(power)
    power_dB -= np.max(power_dB)  # normalize

    return power_dB


def wideband_cssm_mvdr(
    tetrahedra: TetrahedralArray,
    signal: NDArray[np.float64],
    center_freq: float,
    Theta: NDArray[np.float64],
    Phi: NDArray[np.float64],
    initial_doa: NDArray[np.float64] = None,
    num_expected_sources=1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """CSSM MVDR wideband beamformer power computation

    Parameters
    ----------
    tetrahedra : TetrahedralArray
        Tetrahedra used for beamforming
    signal : NDArray
        4 x N signal containing received signal by the four hydrophones
    center_freq : float
        Frequency used to steer the selected frequency bins
    Theta : NDArray[np.float64]
        range of polar angles
    Phi : NDArray[np.float64]
        range of azimutal angles
    initial_doa : NDArray[np.float64], optional
        3x1 vector representing the direction of arrival, by default None
    num_expected_sources : int
        number of expected sources in signal

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        Power in db at each scanned angle, and scanned angles

    Raises
    ------
    ValueError
        Make sure to provide Theta and Phi, or neither to get default grids
    """

    if Theta is None and Phi is None:
        Theta, Phi = _get_theta_phi_coarse()
    elif Theta is not None and Phi is not None:
        pass
    else:
        raise ValueError(
            f"Incorrect argument. We need either Theta and Phi valued at None or both arrays need to be provided."
        )

    if num_expected_sources is None:
        num_expected_sources = estimate_num_sources(tetrahedra, signal)

    R = _compute_cssm_correlation_matrix(
        tetrahedra, signal, center_freq, initial_doa, num_expected_sources
    )

    s = _get_steering_vector(center_freq, tetrahedra, Theta, Phi)
    power_dB = _get_power_mvdr(s, R)

    return power_dB, Theta, Phi


def wideband_cssm_mvdr_optimized(
    tetrahedra: TetrahedralArray,
    signal: NDArray[np.float64],
    center_freq: float,
    initial_doa: NDArray[np.float64] = None,
    num_expected_sources=1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """CSSM MVDR wideband beamformer power computation optimized with coarse and fine search

    Parameters
    ----------
    tetrahedra : TetrahedralArray
        Tetrahedra used for beamforming
    signal : NDArray
        4 x N signal containing received signal by the four hydrophones
    center_freq : float
        Frequency used to steer the selected frequency bins
    initial_doa : NDArray[np.float64], optional
        3x1 vector representing the direction of arrival, by default None
    num_expected_sources : int
        number of expected sources in signal

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        Power in db at each scanned angle, and scanned angles

    """

    Theta, Phi = _get_theta_phi_coarse(n_theta=24, n_phi=48)

    if num_expected_sources is None:
        num_expected_sources = estimate_num_sources(tetrahedra, signal)

    R = _compute_cssm_correlation_matrix(
        tetrahedra, signal, center_freq, initial_doa, num_expected_sources
    )

    s = _get_steering_vector(center_freq, tetrahedra, Theta, Phi)
    power_dB = _get_power_mvdr(s, R)

    u = _doa_from_power(power_dB, Theta, Phi)

    Theta, Phi = _get_theta_phi_fine(u, n_theta=100, n_phi=100)
    s = _get_steering_vector(center_freq, tetrahedra, Theta, Phi)
    power_dB = _get_power_mvdr(s, R)
    return power_dB, Theta, Phi


def wideband_cssm_mvdr_doa(
    tetrahedra,
    signal,
    center_freq,
    Theta=None,
    Phi=None,
    initial_doa=None,
    num_expected_sources=1,
) -> NDArray[np.float64]:
    return _doa_from_power(
        *wideband_cssm_mvdr(
            tetrahedra,
            signal,
            center_freq,
            Theta,
            Phi,
            initial_doa=initial_doa,
            num_expected_sources=num_expected_sources,
        )
    )


def wideband_cssm_mvdr_optimized_doa(
    tetrahedra, signal, center_freq, initial_doa=None, num_expected_sources=1
) -> NDArray[np.float64]:
    return _doa_from_power(
        *wideband_cssm_mvdr_optimized(
            tetrahedra,
            signal,
            center_freq,
            initial_doa=initial_doa,
            num_expected_sources=num_expected_sources,
        )
    )


def get_power_music(
    steering_vector: NDArray[np.complex128],
    correlation_matrix: NDArray[np.complex128],
    num_expected_sources,
) -> NDArray[np.float64]:
    """MUSIC power computation from steering vector and correlation matrix

    Parameters
    ----------
    steering_vector : NDArray[np.complex128]
        Steering vector 4, T, P sized matrix
    correlation_matrix : NDArray[np.complex128]
        correlation matrix 4, 4 matrix used for MVDR algorithm

    Returns
    -------
    NDArray[np.float64]
        Power in db at each scanned angle
    """
    _, T, P = np.shape(steering_vector)
    s_flat = steering_vector.reshape(4, T * P)  # (4, T*P)

    w, v = np.linalg.eig(
        correlation_matrix
    )  # eigenvalue decomposition, v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
    eig_val_order = np.argsort(np.abs(w))  # find order of magnitude of eigenvalues
    v = v[:, eig_val_order]  # sort eigenvectors using this order
    # We make a new eigenvector matrix representing the "noise subspace", it's just the rest of the eigenvalues
    V = np.zeros((4, 4 - num_expected_sources), dtype=np.complex64)
    for i in range(4 - num_expected_sources):
        V[:, i] = v[:, i]

    vvh_s = V @ V.conj().T @ s_flat

    denominator = np.sum(
        s_flat.conj() * vvh_s,
        axis=0,
        keepdims=True,
    )

    metric_flat = 1 / np.maximum(np.abs(denominator), 1e-12)
    metric_flat = 10 * np.log10(metric_flat)
    metric_flat -= np.max(metric_flat)
    metric = metric_flat.reshape(T, P)

    return metric


def wideband_cssm_music(
    tetrahedra: TetrahedralArray,
    signal: NDArray[np.float64],
    center_freq: float,
    Theta: NDArray[np.float64],
    Phi: NDArray[np.float64],
    initial_doa=None,
    num_expected_sources=1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """CSSM MUSIC wideband beamformer power computation

    Parameters
    ----------
    tetrahedra : TetrahedralArray
        Tetrahedra used for beamforming
    signal : NDArray
        4 x N signal containing received signal by the four hydrophones
    center_freq : float
        Frequency used to steer the selected frequency bins
    Theta : NDArray[np.float64]
        range of polar angles
    Phi : NDArray[np.float64]
        range of azimutal angles
    initial_doa : NDArray[np.float64], optional
        3x1 vector representing the direction of arrival, by default None
    num_expected_sources : int
        number of expected sources in signal

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        Power in db at each scanned angle, and scanned angles

    Raises
    ------
    ValueError
        Make sure to provide Theta and Phi, or neither to get default grids
    """

    if Theta is None and Phi is None:
        Theta, Phi = _get_theta_phi_coarse()
    elif Theta is not None and Phi is not None:
        pass
    else:
        raise ValueError(
            f"Incorrect argument. We need either Theta and Phi valued at None or both arrays need to be provided."
        )

    if num_expected_sources is None:
        num_expected_sources = estimate_num_sources(tetrahedra, signal)

    R = _compute_cssm_correlation_matrix(
        tetrahedra, signal, center_freq, initial_doa, num_expected_sources
    )

    s = _get_steering_vector(center_freq, tetrahedra, Theta, Phi)
    metric_dB = get_power_music(s, R, num_expected_sources)

    return metric_dB, Theta, Phi


def wideband_cssm_music_optimized(
    tetrahedra, signal, center_freq, initial_doa=None, num_expected_sources=1
):
    """CSSM MUSIC wideband beamformer power computation optimized with coarse and fine search

    Parameters
    ----------
    tetrahedra : TetrahedralArray
        Tetrahedra used for beamforming
    signal : NDArray
        4 x N signal containing received signal by the four hydrophones
    center_freq : float
        Frequency used to steer the selected frequency bins
    initial_doa : NDArray[np.float64], optional
        3x1 vector representing the direction of arrival, by default None
    num_expected_sources : int
        number of expected sources in signal

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        Power in db at each scanned angle, and scanned angles

    """
    Theta, Phi = _get_theta_phi_coarse(n_theta=24, n_phi=48)

    if num_expected_sources is None:
        num_expected_sources = estimate_num_sources(tetrahedra, signal)

    R = _compute_cssm_correlation_matrix(
        tetrahedra, signal, center_freq, initial_doa, num_expected_sources
    )

    s = _get_steering_vector(center_freq, tetrahedra, Theta, Phi)
    power_dB = _get_power_mvdr(s, R)

    u = _doa_from_power(power_dB, Theta, Phi)
    print("finished coarse")

    Theta, Phi = _get_theta_phi_fine(u, n_theta=100, n_phi=100)

    s = _get_steering_vector(center_freq, tetrahedra, Theta, Phi)
    metric_dB = get_power_music(s, R, num_expected_sources)

    return metric_dB, Theta, Phi


def wideband_cssm_music_doa(
    tetrahedra,
    signal,
    center_freq,
    Theta=None,
    Phi=None,
    initial_doa=None,
    num_expected_sources=1,
) -> NDArray[np.float64]:
    return _doa_from_power(
        *wideband_cssm_music(
            tetrahedra,
            signal,
            center_freq,
            Theta,
            Phi,
            initial_doa=initial_doa,
            num_expected_sources=num_expected_sources,
        )
    )


def wideband_cssm_music_optimized_doa(
    tetrahedra, signal, center_freq, initial_doa=None, num_expected_sources=1
) -> NDArray[np.float64]:
    return _doa_from_power(
        *wideband_cssm_music_optimized(
            tetrahedra,
            signal,
            center_freq,
            initial_doa=initial_doa,
            num_expected_sources=num_expected_sources,
        )
    )
