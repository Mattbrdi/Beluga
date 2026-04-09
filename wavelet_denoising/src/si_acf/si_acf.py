import numpy as np
from pywt import iswt  
import pywt

import numpy.typing as npt

def auto_correlation_k(signal : npt.NDArray[np.float64], k : int) -> np.float64:
    #TODO: check if this is really working
    N = len(signal) - k
    return np.dot(signal[:N], signal[k:])


# def auto_correlation(signal : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
#     return np.array([auto_correlation_k(signal, k) for k in range(len(signal))], dtype=np.float64)


def auto_correlation(signal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    signal = np.asarray(signal, dtype=np.float64)
    n = signal.size

    # zero-pad to avoid circular correlation
    fft_size = 1 << (2 * n - 1).bit_length()

    X = np.fft.rfft(signal, n=fft_size)
    ac = np.fft.irfft(X * np.conj(X), n=fft_size)[:n]

    return ac

def level_determination(fundemental_frequency : float, sampling_frequency : float) -> int:
    return int(np.log2(sampling_frequency / fundemental_frequency)) - 1

def _plot_rho_debug(
    auto_correlation: npt.NDArray[np.float64],
    zero_thr: float,
    m: int,
    k: int,
    title: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(len(auto_correlation))
    plt.figure(figsize=(10, 4))
    plt.plot(x, auto_correlation, label="autocorrelation")
    plt.axhline(zero_thr, color="tab:red", linestyle="--", linewidth=1, label="zero_thr")
    plt.axhline(-zero_thr, color="tab:red", linestyle="--", linewidth=1)
    plt.axvline(m, color="tab:orange", linestyle=":", linewidth=1, label="first near-zero")
    plt.scatter([k], [auto_correlation[k]], color="tab:green", zorder=3, label="selected peak")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title(title or "rho debug")
    plt.legend()
    plt.tight_layout()
    plt.show()


def rho(auto_correlation : npt.NDArray[np.float64], zero_thr=1e-3, debug_plot = False) -> np.float64:
    R0 = auto_correlation[0]
    # print("auto_correlation", np.shape(auto_correlation))
    if R0 == 0:
        return 1
    
    #TODO: verify and understand this function (is it m or m -1)
    # idx = np.where(np.abs(auto_correlation) <= zero_thr)[0]
    idx = np.where(np.diff(np.sign(auto_correlation)))[0]
    m = idx[0] if idx.size > 0 else np.argmax(auto_correlation[1:])
    # print("m", m)
    k = np.argmax(auto_correlation[m+1:]) + (m + 1) 

    if debug_plot:
            _plot_rho_debug(
                auto_correlation=auto_correlation,
                zero_thr=zero_thr,
                m=m,
                k=k,
                title="auto_correlation viewer",
            )

    if k == len(auto_correlation) - 1:
        return 0 # est-ce-que c'est ce threshold ?  
    return auto_correlation[k] / R0

def compute_P1(all_coeffs : npt.NDArray[np.float64], coeffs : np.ndarray, wavelet : pywt.Wavelet) -> npt.NDArray[np.float64]:
    #assuming coeffs are in the format (AL, DL, ..., D1)
    original_shape = np.shape(coeffs)
    new_shape = (original_shape[0], original_shape[0], original_shape[1])

    i = np.arange(original_shape[0])

    individual_coeffs = np.zeros(shape=new_shape, dtype=np.float64)
    individual_coeffs[i,i,:] = coeffs
    
    def format_coeff(i, coeff): 
        length = original_shape[0]
        array = [(0, 0)] * length
        if i == 0:
            array[i] = (coeff, 0)
        else:
            array[i] = (0, coeff)
        return np.array(array)

    # individual_signals = np.array([iswt(individual_coeff, wavelet) for individual_coeff in individual_coeffs])
    individual_signals =  np.array([iswt(format_coeff(i, c), wavelet) for i, c in enumerate(individual_coeffs)])

    auto_correlations = np.array([auto_correlation(s) for s in individual_signals])

    rhos = np.array([rho(ac) for ac in auto_correlations], dtype=np.float64)
    return rhos # (rhoAL, rho DL, ... rho D1)

def compute_P2(all_coeffs : npt.NDArray[np.float64], coeffs : npt.NDArray[np.float64], wavelet : pywt.Wavelet) -> npt.NDArray[np.float64]:
    #assuming coeffs are in the format (AL, DL, ..., D1)
    original_shape = np.shape(coeffs)
    new_shape = (original_shape[0], original_shape[0], original_shape[1])
    i = np.arange(original_shape[0])
    new_coeffs = individual_coeffs = np.zeros(shape=new_shape, dtype=np.float64)
    new_coeffs[i,:,:] = coeffs.copy()
    new_coeffs[i,i,:] = np.zeros(shape=new_shape[2], dtype=np.float64)
    
    # print(np.shape(coeffs))
    # print("uwu",np.shape(new_coeffs))

    
    def format_coeff(i, coeff): 
        length = original_shape[0]
        array = [(0, 0)] * length
        if i == 0:
            array[i] = (coeff, 0)
        else:
            array[i] = (0, coeff)
        return np.array(array)

    # individual_signals = np.array([iswt(individual_coeff, wavelet) for individual_coeff in individual_coeffs])
    individual_signals =  np.array([iswt(format_coeff(i, c), wavelet) for i, c in enumerate(individual_coeffs)])

    auto_correlations = np.array([auto_correlation(s) for s in individual_signals])

    rhos = np.array([rho(ac) for ac in auto_correlations])

    return rhos

def rho_thr(all_coeffs : npt.NDArray[np.float64], coeffs : npt.NDArray[np.float64], wavelet : pywt.Wavelet, level : int, thr : float) -> npt.NDArray[np.float64]:
    new_coeffs = np.copy(coeffs)
    new_coeffs[level][np.abs(new_coeffs[level]) < thr] = 0

    all_new_coeffs = [(ca, cd) for (ca, cd) in zip(all_coeffs[::2], coeffs[1:])]

    new_signal = iswt(all_new_coeffs, wavelet)

    auto_corr = auto_correlation(new_signal)
    return rho(auto_corr)

# TODO: recheck, add comment, check if working
def trisection(current_coeffs : np.ndarray, level : int, wavelet : pywt.Wavelet, tol = 1e-6) -> float:
    D_i = current_coeffs[level]
    ak = 0.0
    dk = float(np.max(np.abs(D_i)))

    rho_ak = rho_thr(current_coeffs, wavelet, level, ak)
    rho_dk = rho_thr(current_coeffs, wavelet, level, dk)

    while abs(ak - dk) >= tol:
        bk = ak + (dk - ak) / 3.0
        ck = dk - (dk - ak) / 3.0

        rho_bk = rho_thr(current_coeffs, wavelet, level, bk)
        rho_ck = rho_thr(current_coeffs, wavelet, level, ck)
        
        rhos = np.array([rho_ak, rho_bk, rho_ck, rho_dk])
        argmax = int(np.argmax(rhos))
        if argmax == 0:
            dk = bk
            rho_dk = rho_bk
        elif argmax == 1:
            dk = ck
            rho_dk = rho_ck
        elif argmax == 2:
            ak = bk
            rho_ak = rho_bk
        elif argmax == 3:
            ak = ck
            rho_ak = rho_ck
    return ak if rho_ak > rho_dk else dk     


def non_impulsive_noise_filter(signal : npt.NDArray[np.float64], coeffs : npt.NDArray[np.float64], wavelet : pywt.Wavelet, fs : float) -> npt.NDArray[np.float64]:
    # print("shape debug", np.shape(coeffs))
    details_coeffs = [cd for (ca, cd) in coeffs]
    # print(np.shape(details_coeffs))
    l = np.size(details_coeffs)

    AL = coeffs[0][0]

    details_coeffs = np.array([AL, *details_coeffs])
    P1 = compute_P1(coeffs, details_coeffs, wavelet)

    P2 = compute_P2(coeffs, details_coeffs, wavelet)

    # Compute w: the whistle signal most correlated component:
    w = np.argmax(P1[1:]) + 1# reversed P1 looks like (D1, ... DL, AL)
    # print("w", w)
    new_coeffs = np.copy(details_coeffs)

    if l != (l + 1 - w) and (l + 1 - w) != l - 1 and P2[0] != np.argmin(P2):
        AL = np.zeros(shape=np.shape(AL))
        new_coeffs[0] = AL

    for i in range(1, len(new_coeffs)):
        if i != w:
            thr = trisection(new_coeffs, i, wavelet)
            new_coeffs[i][np.abs(new_coeffs[i]) < thr] = 0

    win = int(0.1 * fs)
    D_w = new_coeffs[w]
    a = np.abs(D_w)
    n = (a.size // win) * win
    if n == 0:
        raise ValueError("window larger than array")
    sep_signal = a[:n].reshape(-1, win)
    window_means = sep_signal.mean(axis=1)

    M_min = window_means.min()
    M_mean = window_means.mean()

    MH = M_mean - (M_mean - M_min) / 2
    ML = M_min / 2

    #TODO: find how to properly rewrite this
    ak = ML
    dk = MH

    rho_ak = rho_thr(coeffs, new_coeffs, wavelet, w, ak)
    rho_dk = rho_thr(coeffs, new_coeffs, wavelet, w, dk)

    while abs(ak - dk) >= 1e-6:
        bk = ak + (dk - ak) / 3.0
        ck = dk - (dk - ak) / 3.0

        rho_bk = rho_thr(coeffs, new_coeffs, wavelet, w, bk)
        rho_ck = rho_thr(coeffs, new_coeffs, wavelet, w, ck)
        
        rhos = np.array([rho_ak, rho_bk, rho_ck, rho_dk])
        argmax = int(np.argmax(rhos))
        if argmax == 0:
            dk = bk
            rho_dk = rho_bk
        elif argmax == 1:
            dk = ck
            rho_dk = rho_ck
        elif argmax == 2:
            ak = bk
            rho_ak = rho_bk
        elif argmax == 3:
            ak = ck
            rho_ak = rho_ck
    thr = ak if rho_ak > rho_dk else dk

    new_coeffs[w][np.abs(new_coeffs[w]) < thr] = 0

    #TODO: verify what to return  
    return new_coeffs
  

    

