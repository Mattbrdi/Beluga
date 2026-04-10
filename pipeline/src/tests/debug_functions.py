import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, windows

def analyse_signal(signal, fs, seuil_band_db=-15, seuil_peaks=0.2, frequency_range = (500,2000)):
    """Analyse un seul canal et retourne infos spectrales + SNR"""
    N = len(signal)
    window = windows.hann(N)
    spectrum = np.fft.rfft(signal * window)
    freqs = np.fft.rfftfreq(N, 1./fs)
    power = np.abs(spectrum)**2

    # --- Ignorer les fréquences < fmin si demandé ---
    start_idx = np.searchsorted(freqs, frequency_range[0])
    end_idx = np.searchsorted(freqs,frequency_range[1])
    freqs = freqs[start_idx:end_idx]
    power = power[start_idx:end_idx]

    # --- Détection pics ---
    peaks, _ = find_peaks(power, height=np.max(power)*seuil_peaks)
    peak_freqs = freqs[peaks]

    # --- Bande utile (seuil relatif) ---
    threshold = np.max(power) * (10**(seuil_band_db/10))
    indices = np.where(power >= threshold)[0]
    band = (freqs[indices[0]], freqs[indices[-1]])

    # --- Fréquence centrale ---
    f_centrale = np.sum(freqs * power) / np.sum(power)

    # --- SNR (bande utile vs hors bande) ---
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    signal_power = np.sum(power[band_mask]) / np.sum(band_mask)
    noise_power = np.sum(power[~band_mask]) / np.sum(~band_mask)
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-20))
    
    # --- Bande passante autour de f_centrale (-3 dB) ---
    idx_center = np.argmin(np.abs(freqs - f_centrale))
    center_power = power[idx_center]
    half_power = center_power / 2  # -3 dB
    # gauche
    left_idx = np.where(power[:idx_center] <= half_power)[0]
    left_bound = freqs[left_idx[-1]] if len(left_idx) else freqs[0]
    # droite
    right_idx = np.where(power[idx_center:] <= half_power)[0]
    right_bound = freqs[idx_center + right_idx[0]] if len(right_idx) else freqs[-1]
    bp_centrale = right_bound - left_bound
    
    bande_utile_peaks = (peak_freqs[0], peak_freqs[-1])
    band_mask_peaks = (freqs >= bande_utile_peaks[0]) & (freqs <= bande_utile_peaks[1])
    signal_power_peaks = np.sum(power[band_mask_peaks]) / np.sum(band_mask_peaks)
    noise_power_peaks = np.sum(power[~band_mask_peaks]) / np.sum(~band_mask_peaks)
    snr_db_peaks = 10*np.log10(signal_power_peaks / (noise_power_peaks)+1E-20)

    return {
        "freq_centrale": f_centrale,
        "band": band,
        "bp_centrale" : bp_centrale,
        "peaks": peak_freqs,
        "snr_db": snr_db,
        "freqs": freqs,
        "power": power,
        "peaks_idx": peaks,
        "snr_db_peaks" : snr_db_peaks,
        "peaks_band" : bande_utile_peaks
    }


import numpy as np
import matplotlib.pyplot as plt

def plot_analysis(signals, sample_rate, frequency_range):
    results = []

    fig, axes = plt.subplots(len(signals), 1, figsize=(10, 12))
    if len(signals) == 1:
        axes = [axes]  # un seul canal → liste unique

    for i, (sig, ax) in enumerate(zip(signals, axes)):
        # --- Analyse du canal ---
        res = analyse_signal(sig, sample_rate,frequency_range=frequency_range)
        results.append(res)
        ax.axvline(x = res["band"][0], color='r', linestyle='--')
        ax.axvline(x = res["band"][1], color='r', linestyle='--')
        
        # Ajout peaks analysis
        ax.axvline(x = res["peaks_band"][0], color='green', linestyle='--')
        ax.axvline(x = res["peaks_band"][1], color='green', linestyle='--')

        # --- Tracé du spectre (échelle linéaire) ---
        ax.plot(res["freqs"], res["power"], label=f'Canal {i+1}')
        ax.plot(res["freqs"][res["peaks_idx"]], res["power"][res["peaks_idx"]], "rx")
        # Bande passante du pic dominant (trait vertical)
        # --- Annotations texte ---
        text = (f"f_centrale = {res['freq_centrale']:.1f} Hz\n"
                f"Bande = {res['band'][0]:.1f}-{res['band'][1]:.1f} Hz\n"
                f"Bande from peaks {res['peaks_band'][0]:.1f}-{res['peaks_band'][1]:.1f} Hz\n"
                f"SNR = {res['snr_db']:.1f} dB\n"
                f"SNR from peaks = {res['snr_db_peaks']:.1f} dB")
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.set_xlabel("Fréquence (Hz)")
        ax.set_ylabel("Puissance (linéaire)")
        ax.set_title(f"Spectre Canal {i+1}")
        ax.legend()

    plt.tight_layout()
    plt.show()

