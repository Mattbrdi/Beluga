import numpy as np
from pywt import iswt  
import pywt

import numpy.typing as npt

import ruptures as rpt

def full_swt_coeffs_to_approx_coeff(full_swt_coeffs : npt.NDArray[np.float64]):
    """! Extracts the SWT coefficients into the reduced format that keeps
    only the top-level approximation coefficient and all detail coefficients.

    @param full_swt_coeffs The SWT coefficients in the format
                           (A_L, D_L), ..., (A_1, D_1)

    @return The coefficients in the format A_L, D_L, ..., D_1
    """    
    final_approx_coeffs = [cd for (ca, cd) in full_swt_coeffs] # get (DL, ..., D_1)
    AL = full_swt_coeffs[0][0] 
    return np.array([AL, *final_approx_coeffs])

def replace_approx_coeff_in_full_swt_coeffs(full_swt_coeffs : npt.NDArray[np.float64], final_approx_coeffs : npt.NDArray[np.float64]):
    """! Replaces the top-level approximation coefficient and the detail
    coefficients in a full SWT coefficient structure.

    @param full_swt_coeffs The SWT coefficients in the format
                        (A_L, D_L), ..., (A_1, D_1)

    @param final_approx_coeffs The coefficients in the format
                            A_L, D_L, ..., D_1

    @return A copy of the SWT coefficients in the format
            (A_L, D_L), ..., (A_1, D_1) with the updated values
    """
    new_full_swt_coeffs = [(ca.copy(), cd.copy()) for ca, cd in full_swt_coeffs]

    # Replace detail coeffs
    new_full_swt_coeffs = [
        (ca, new_cd)
        for (ca, _old_cd), new_cd in zip(new_full_swt_coeffs, final_approx_coeffs[1:])
    ]

    # Replace top-level approximation coeffs
    new_full_swt_coeffs[0] = (final_approx_coeffs[0], new_full_swt_coeffs[0][1])

    return [(ca, cd) for ca, cd in new_full_swt_coeffs]

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
    if R0 == 0:
        return 1
    
    #TODO: verify and understand this function (is it m or m -1)
    # idx = np.where(np.abs(auto_correlation) <= zero_thr)[0]
    idx = np.where(np.diff(np.sign(auto_correlation)))[0]
    m = idx[0] if idx.size > 0 else np.argmax(auto_correlation[1:]) + 1 
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

def compute_P1(full_swt_coeffs : npt.NDArray[np.float64], wavelet : pywt.Wavelet) -> npt.NDArray[np.float64]:
    #assuming coeffs are in the format (AL, DL, ..., D1)

    full_approx_coeffs = full_swt_coeffs_to_approx_coeff(full_swt_coeffs)

    original_shape = np.shape(full_approx_coeffs)
    new_shape = (original_shape[0], original_shape[0], original_shape[1])

    i = np.arange(original_shape[0])

    individual_coeffs = np.zeros(shape=new_shape, dtype=np.float64)
    individual_coeffs[i,i,:] = full_approx_coeffs

    individual_signals =  np.array([iswt(replace_approx_coeff_in_full_swt_coeffs(full_swt_coeffs, ac), wavelet) for ac in individual_coeffs])
    auto_correlations = np.array([auto_correlation(s) for s in individual_signals])
    rhos = np.array([rho(ac) for ac in auto_correlations], dtype=np.float64)
    return rhos # (rhoAL, rho DL, ... rho D1)

def compute_P2(full_swt_coeffs : npt.NDArray[np.float64], wavelet : pywt.Wavelet) -> npt.NDArray[np.float64]:
    #assuming coeffs are in the format (AL, DL, ..., D1)*

    full_approx_coeffs = full_swt_coeffs_to_approx_coeff(full_swt_coeffs)

    original_shape = np.shape(full_approx_coeffs)
    new_shape = (original_shape[0], original_shape[0], original_shape[1])
    i = np.arange(original_shape[0])
    new_coeffs = individual_coeffs = np.zeros(shape=new_shape, dtype=np.float64)
    new_coeffs[i,:,:] = full_approx_coeffs.copy()
    new_coeffs[i,i,:] = np.zeros(shape=new_shape[2], dtype=np.float64)
    
    individual_signals =  np.array([iswt(replace_approx_coeff_in_full_swt_coeffs(full_swt_coeffs, ac), wavelet) for ac in individual_coeffs])

    auto_correlations = np.array([auto_correlation(s) for s in individual_signals])

    rhos = np.array([rho(ac) for ac in auto_correlations])

    return rhos

def rho_thr(full_swt_coeffs : npt.NDArray[np.float64], wavelet : pywt.Wavelet, level : int, thr : float) -> npt.NDArray[np.float64]:

    full_approx_coeffs = full_swt_coeffs_to_approx_coeff(full_swt_coeffs)

    new_coeffs = np.copy(full_approx_coeffs)
    new_coeffs[level][np.abs(new_coeffs[level]) < thr] = 0

    all_new_coeffs = replace_approx_coeff_in_full_swt_coeffs(full_swt_coeffs, new_coeffs)

    new_signal = iswt(all_new_coeffs, wavelet)

    auto_corr = auto_correlation(new_signal)
    return rho(auto_corr)

# TODO: recheck, add comment, check if working
def trisection(full_swt_coeffs : np.ndarray, level : int, wavelet : pywt.Wavelet, tol = 1e-6) -> float:
    current_coeffs = full_swt_coeffs_to_approx_coeff(full_swt_coeffs)
    D_i = current_coeffs[level]
    ak = 0.0
    dk = float(np.max(np.abs(D_i)))
    rho_ak = rho_thr(full_swt_coeffs, wavelet, level, ak)
    rho_dk = rho_thr(full_swt_coeffs, wavelet, level, dk)

    while abs(ak - dk) >= tol:
        bk = ak + (dk - ak) / 3.0
        ck = dk - (dk - ak) / 3.0

        rho_bk = rho_thr(full_swt_coeffs, wavelet, level, bk)
        rho_ck = rho_thr(full_swt_coeffs, wavelet, level, ck)
        
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
    details_coeffs = full_swt_coeffs_to_approx_coeff(coeffs)
    l = len(details_coeffs) - 1

    P1 = compute_P1(coeffs, wavelet)

    P2 = compute_P2(coeffs, wavelet)

    # Compute w: the whistle signal most correlated component:
    w = np.argmax(P1[1:]) + 1# reversed P1 looks like (D1, ... DL, AL)
    new_coeffs = np.copy(details_coeffs)

    AL = new_coeffs[0]

    if l != (l + 1 - w) and (l + 1 - w) != l - 1 and P2[0] != np.argmin(P2):
        AL = np.zeros(shape=np.shape(AL))
        new_coeffs[0] = AL

    new_full_swt_coeffs = coeffs.copy()

    for i in range(1, len(new_coeffs)):
        if i != w:
            new_full_swt_coeffs = replace_approx_coeff_in_full_swt_coeffs(new_full_swt_coeffs, new_coeffs)
            print(np.shape(new_full_swt_coeffs))
            thr = trisection(new_full_swt_coeffs, i, wavelet)
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

    new_full_swt_coeffs = replace_approx_coeff_in_full_swt_coeffs(coeffs, new_coeffs) 

    rho_ak = rho_thr(new_full_swt_coeffs, wavelet, w, ak)
    rho_dk = rho_thr(new_full_swt_coeffs, wavelet, w, dk)

    while abs(ak - dk) >= 1e-6:
        bk = ak + (dk - ak) / 3.0
        ck = dk - (dk - ak) / 3.0

        rho_bk = rho_thr(new_full_swt_coeffs, wavelet, w, bk)
        rho_ck = rho_thr(new_full_swt_coeffs, wavelet, w, ck)
        
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
    return replace_approx_coeff_in_full_swt_coeffs(coeffs, new_coeffs)
  
def impulsive_noise_filter(signal : npt.NDArray[np.float64], coeffs : npt.NDArray[np.float64], wavelet : pywt.Wavelet, fs : float) -> npt.NDArray[np.float64]:
    window_length = int(0.1 * fs)
    signal_length = len(signal)
    full_approx_coeff = full_swt_coeffs_to_approx_coeff(coeffs)    

    K = 1

    for coeff in full_approx_coeff[1:]:
        for i in range(0,signal_length - window_length + 1, window_length // 2):
            chunk = coeff[i:i + window_length].copy()
            sorted_indexes = np.argsort(chunk)
            sorted_chunk = chunk[sorted_indexes]

            positive_sorted_chunk = sorted_chunk[sorted_chunk > 0]
            negative_sorted_chunk = sorted_chunk[sorted_chunk <= 0]

            bkps_neg = None
            changepoints_neg = None
            bkps_pos = None
            changepoints_pos = None

            # positive side
            if len(positive_sorted_chunk) >= 4:
                algo_pos = rpt.Dynp(model="normal", min_size=2).fit(positive_sorted_chunk)
                bkps_pos = algo_pos.predict(n_bkps=K)
                cp = bkps_pos[0]
                print("cp", cp, i + sorted_indexes[cp])
                if 0 < cp < len(positive_sorted_chunk):
                    t_plus = positive_sorted_chunk[cp - 1]
                    mu_plus = np.mean(positive_sorted_chunk[cp:])
                    chunk[chunk > t_plus] = mu_plus

            # negative side
            if len(negative_sorted_chunk) >= 4:
                algo_neg = rpt.Dynp(model="normal", min_size=2).fit(negative_sorted_chunk)
                bkps_neg = algo_neg.predict(n_bkps=K)
                cp = bkps_neg[0]

                if 0 < cp < len(negative_sorted_chunk):
                    t_minus = negative_sorted_chunk[cp]
                    mu_neg = np.mean(negative_sorted_chunk[:cp])
                    chunk[chunk < t_minus] = mu_neg

            coeff[i: i + window_length] = chunk
    
    return replace_approx_coeff_in_full_swt_coeffs(coeffs, full_approx_coeff)
