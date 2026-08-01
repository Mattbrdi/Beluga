import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
from scipy.signal import hilbert

from beamforming.configuration import *
from beamforming.classes import *


def _get_theta_phi_coarse(n_theta=50, n_phi=50):
    """Get the angles search grid spanning the whole sphere

    Parameters
    ----------
    n_theta : int, optional
        Number of points in the polar grid, by default 500
    n_phi : int, optional
        Number of points in the azimuthal grid, by default 500

    Returns
    -------
    NDArray
        Returns the coarse angles grid
    """
    theta_scan = np.linspace(0, np.pi, n_theta)
    phi_scan = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    return np.meshgrid(theta_scan, phi_scan, indexing="ij")  # T x P

def _get_steering_vector(fc, tetrahedra, Theta, Phi):
    p12 = tetrahedra.p2 - tetrahedra.p1
    p13 = tetrahedra.p3 - tetrahedra.p1
    p14 = tetrahedra.p4 - tetrahedra.p1    

    k = np.array(
        [
            np.sin(Theta) * np.cos(Phi),
            np.sin(Theta) * np.sin(Phi),
            np.cos(Theta),
        ],
        dtype=np.float64,
    )  # (3, T, P)

    kdotp12 = np.tensordot(k, p12, axes=(0, 0))  # (T, P)
    kdotp13 = np.tensordot(k, p13, axes=(0, 0))  # (T, P)
    kdotp14 = np.tensordot(k, p14, axes=(0, 0))  # (T, P)

    s = np.stack(
        [
            np.ones_like(kdotp12, dtype=np.complex128),
            np.exp(2j * np.pi * fc * kdotp12 / C),
            np.exp(2j * np.pi * fc * kdotp13 / C),
            np.exp(2j * np.pi * fc * kdotp14 / C),
        ],
        axis=0,
    )
    return s

def _doa_from_power(power_dB, Theta, Phi):
    theta_idx, phi_idx = np.unravel_index(
        np.argmax(power_dB),
        power_dB.shape,
    )
    theta, phi = Theta[theta_idx, phi_idx], Phi[theta_idx, phi_idx]
    return np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    ).astype(np.float64)



def delay_and_sum(
    fc: int, tetrahedra: TetrahedralArray, signal: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Delay and Sum Beamformer power computation for narrowband signals

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
    max_power = 0
    n_per_chunk = 1500

    Theta, Phi = _get_theta_phi_coarse()
    s = _get_steering_vector(fc, tetrahedra, Theta, Phi)
    T, P = Theta.shape
    s_flat = s.reshape(4, T * P)  # (4, T*P)

    power_flat = np.ones(shape=(T * P), dtype=np.float32)

    N = int(np.ceil(s_flat.shape[1] / n_per_chunk))
    print(N)
    for i in range(N):
        print(f"{i} / {N}")
        idx_min = i * n_per_chunk
        idx_max = min(s_flat.shape[1], (i + 1) * n_per_chunk)

        s_flat_chunk = s_flat[:, idx_min:idx_max].astype(np.complex64)

        X_weighted_flat_chunk = s_flat_chunk.conj().T @  hilbert(signal, axis=1)
        power_flat[idx_min:idx_max] = np.mean(
            np.abs(X_weighted_flat_chunk) ** 2,
            axis=1,
        )

    power = power_flat.reshape(T, P)

    power_dB = 10 * np.log10(power)

    power_dB -= np.max(power_dB)  # normalize

    return power_dB, Theta, Phi


def delay_and_sum_doa(
    fc: int, tetrahedra: TetrahedralArray, signal: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Delay and Sum Beamformer for DOA finding of narrowband signals

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
    return _doa_from_power(*delay_and_sum(fc, tetrahedra, signal))


def mvdr(
    fc: float, tetrahedra: TetrahedralArray, signal: NDArray[np.float64]
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
    Theta, Phi = _get_theta_phi_coarse()
    s = _get_steering_vector(fc, tetrahedra, Theta, Phi)
    T, P = Theta.shape
    s_flat = s.reshape(4, T * P)  # (4, T*P)

    x = hilbert(signal, axis=1)

    R = x @ x.conj().T / x.shape[1]
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

    # w_flat = Rinv_s / denominator                         # (4, T*P)
    # # w = w_flat.reshape(4, T, P)                           # (4, T, P)

    # X_weighted_flat = w_flat.conj().T @ signal    # (T*P, Nsamples)

    # X_weighted = X_weighted_flat.reshape(
    #     T, P, signal.shape[1]
    # )
    power_dB = 10 * np.log10(power)

    power_dB -= np.max(power_dB)  # normalize

    return power_dB, Theta, Phi


def mvdr_doa(
    fc: float, tetrahedra: TetrahedralArray, signal: NDArray[np.float64]
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
    return _doa_from_power(*mvdr(fc, tetrahedra, signal))


def music(
    fc: float,
    tetrahedra: TetrahedralArray,
    signal: NDArray[np.float64],
    num_expected_signals=1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """MUSIC Beamformer for power computation of narrowband signals

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
    Theta, Phi = _get_theta_phi_coarse()
    s = _get_steering_vector(fc, tetrahedra, Theta, Phi)
    T, P = Theta.shape
    s_flat = s.reshape(4, T * P)  # (4, T*P)

    x = hilbert(signal, axis=1)
    R = x @ x.conj().T / x.shape[1]

    # part that doesn't change with theta_i
    # R = np.cov(signal)  # Calc covariance matrix. gives a Nr x Nr covariance matrix
    w, v = np.linalg.eig(
        R
    )  # eigenvalue decomposition, v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
    eig_val_order = np.argsort(np.abs(w))  # find order of magnitude of eigenvalues
    v = v[:, eig_val_order]  # sort eigenvectors using this order
    # We make a new eigenvector matrix representing the "noise subspace", it's just the rest of the eigenvalues
    V = np.zeros((4, 4 - num_expected_signals), dtype=np.complex64)
    for i in range(4 - num_expected_signals):
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

    return metric, Theta, Phi


def music_doa(
    fc: float,
    tetrahedra: TetrahedralArray,
    signal: NDArray[np.float64],
    num_expected_signals=1,
) -> NDArray[np.float64]:
    """MUSIC Beamformer for DOA finding of narrowband signals

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
    return _doa_from_power(
        *music(fc, tetrahedra, signal, num_expected_signals=num_expected_signals)
    )
