import numpy as np
from numpy.typing import NDArray
from scipy.signal import stft

from beamforming.configuration import *
from beamforming.classes import *
from beamforming.mathematics.beamformer import (
    _doa_from_power,
    _get_steering_vector,
    _get_theta_phi_coarse,
    estimate_num_sources,
    _get_steering_vector_per_freq,
)

from time_frequency_mask.configuration import N_FFT, HOP_LENGTH, MIN_FREQ, MAX_FREQ


def tops(
    tetrahedra: TetrahedralArray,
    signal: NDArray[np.float64],
    center_freq: float = None,
    tf_mask: NDArray[np.uint8] = None,
    num_expected_sources: int = 1,
):
    """TOPS wideband beamformer power computation

    Parameters
    ----------
    tetrahedra : TetrahedralArray
        Tetrahedra used for beamforming
    signal : NDArray
        4 x N signal containing received signal by the four hydrophones
    center_freq : float
        Frequency used to steer the selected frequency bins
    tf_mask : NDArray[np.uint8], optional
        MASK on stft
    num_expected_sources : int
        number of expected sources in signal

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        Power in db at each scanned angle, and scanned angles

    """
    Theta, Phi = _get_theta_phi_coarse(n_theta=100, n_phi=100)
    T, P = Theta.shape

    if num_expected_sources is None:
        num_expected_sources = estimate_num_sources(tetrahedra, signal)

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
    freqs, Zxx = freqs[freq_mask], Zxx[:, freq_mask, :]  # F and M, F, Ti

    if tf_mask is not None:
        Zxx = Zxx * (tf_mask != 0)[None, :, :]
        valid_freqs = np.where(tf_mask.sum(axis=1) != 0)[0]
        freqs = freqs[valid_freqs]
        Zxx = Zxx[:, valid_freqs, :]
        tf_mask = tf_mask[valid_freqs, :]

    bin_energy = np.sum(np.abs(Zxx) ** 2, axis=(0, 2))
    f0_idx = np.argmax(bin_energy)
    if center_freq is not None:
        requested_idx = np.argmin(np.abs(freqs - center_freq))

        if bin_energy[requested_idx] > 1e-2 * bin_energy[f0_idx]:
            f0_idx = requested_idx

    M, F, Ti = Zxx.shape
    S = num_expected_sources
    N = M - S

    Zxx = np.swapaxes(Zxx, 0, 1)  # (F, M, Ti)

    R_ks = np.einsum("fmt,fnt->fmn", Zxx, Zxx.conj())  # (F, M, M)

    eigvals, eigvecs = np.linalg.eigh(R_ks)  # (F, M, M)

    signal_subspace = eigvecs[..., -S:][f0_idx]  # (F, M, S)
    noise_subspace = eigvecs[..., :N]  # (F, M, M - S)

    s_ks = _get_steering_vector_per_freq(freqs, tetrahedra, Theta, Phi)  # (F, M, T, P)
    s_0 = _get_steering_vector(freqs[f0_idx], tetrahedra, Theta, Phi)  # (M, T, P)

    eye = np.eye(M, dtype=s_ks.dtype)[None, :, :, None, None]  # (1, M, M, 1, 1)
    steering_norm = np.einsum("fmtp,fmtp->ftp", s_ks.conj(), s_ks)  # (F, T, P)

    Proj = (
        eye
        - np.einsum("fmtp,fntp->fmntp", s_ks, s_ks.conj())
        / steering_norm[:, None, None, :, :]
    )  # (F, M, M, T, P)

    focus_ratio = s_ks / s_0[None, :, :, :]  # (F, M, T, P)
    Focus = eye * focus_ratio[:, :, None, :, :]  # (F, M, M, T, P)

    ProjFocus = np.einsum("fijtp,fjktp->fiktp", Proj, Focus)  # (F, M, M, T, P)
    W = np.einsum("fijtp, js->fistp", ProjFocus, signal_subspace)  # (F, M, S, T, P)
    D = np.einsum("fmstp,fmn->fsntp", W.conj(), noise_subspace)  # (F, S, N, T, P)

    D = D.transpose(1, 0, 2, 3, 4).reshape(S, F * N, T, P)  # (S, F * N, T, P)
    D_batch = D.transpose(2, 3, 0, 1)  # (T, P, S, F * N)

    singular_values = np.linalg.svdvals(D_batch)  # (T, P, S)
    sigma_min = singular_values[..., -1]  # (T, P)

    tops_spectrum = 1.0 / np.maximum(sigma_min, 1e-12)

    tops_spectrum_db = 10 * np.log10(tops_spectrum)
    tops_spectrum_db -= np.max(tops_spectrum_db)

    return tops_spectrum_db, Theta, Phi


def srp_phat(
    tetrahedra: TetrahedralArray,
    signal: NDArray[np.float64],
    tf_mask=None,
    num_expected_sources=1,
):
    """TOPS wideband beamformer power computation

    Parameters
    ----------
    tetrahedra : TetrahedralArray
        Tetrahedra used for beamforming
    signal : NDArray
        4 x N signal containing received signal by the four hydrophones
    tf_mask : NDArray[np.uint8], optional
        MASK on stft
    num_expected_sources : int
        number of expected sources in signal

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        Power in db at each scanned angle, and scanned angles

    """
    Theta, Phi = _get_theta_phi_coarse(n_theta=200, n_phi=200)
    T, P = Theta.shape

    if num_expected_sources is None:
        num_expected_sources = estimate_num_sources(tetrahedra, signal)

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
    freqs, Zxx = freqs[freq_mask], Zxx[:, freq_mask, :]  # F and M, F, Ti

    if tf_mask is not None:
        weights = (tf_mask != 0).astype(float)
    else:
        tf_mask = np.ones((len(freqs), Zxx.shape[-1]), dtype=float)
        weights = tf_mask
    M, F, Ti = Zxx.shape

    Zxx = Zxx.transpose(1, 0, 2)  # (F, M, Ti)

    s_ks = _get_steering_vector_per_freq(freqs, tetrahedra, Theta, Phi)  # (F, M, T, P)

    i, j = np.triu_indices(M, k=1)
    Q = len(i)
    cross_ftq = (Zxx[:, i, :] * Zxx[:, j, :].conj()).transpose(0, 2, 1)  # (F, Ti, Q)

    cross_phat_ftq = np.divide(
        cross_ftq,
        np.abs(cross_ftq),
        out=np.zeros_like(cross_ftq),
        where=np.abs(cross_ftq) > 1e-12,
    )

    weights_sum = weights.sum(axis=1)  # (F,)

    cross_phat_fq = np.einsum(
        "ftq,ft->fq",
        cross_phat_ftq,
        weights,
        optimize=True,
    )  # (F, Q)

    cross_phat_fq = np.divide(
        cross_phat_fq,
        weights_sum[:, None],
        out=np.zeros_like(cross_phat_fq),
        where=weights_sum[:, None] > 0,
    )
    # (F, Q)

    pair_steering = s_ks[:, i, :, :] * s_ks[:, j, :, :].conj()  #  # (F, Q, T, P)

    srp_phat = np.real(
        np.einsum("fq,fqtp->tp", cross_phat_fq, pair_steering.conj(), optimize=True)
    )

    return srp_phat, Theta, Phi


def tops_doa(
    tetrahedra,
    signal,
    center_freq,
    tf_mask=None,
    num_expected_sources=1,
) -> NDArray[np.float64]:
    return _doa_from_power(
        *tops(
            tetrahedra=tetrahedra,
            signal=signal,
            center_freq=center_freq,
            tf_mask=tf_mask,
            num_expected_sources=num_expected_sources,
        )
    )


def srp_phat_doa(
    tetrahedra,
    signal,
    tf_mask=None,
    num_expected_sources=1,
) -> NDArray[np.float64]:
    return _doa_from_power(
        *srp_phat(
            tetrahedra=tetrahedra,
            signal=signal,
            tf_mask=tf_mask,
            num_expected_sources=num_expected_sources,
        )
    )
