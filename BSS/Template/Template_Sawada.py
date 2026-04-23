import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal
from ..signal_class import Signal, MultiSignal, Mixture
from ..Sawada_separation import SawadaBSS, StftParameters

def test_sawada_separation():
    # --- 1. Paramètres de simulation ---
    fs = 8000
    duration = 2.0
    
    # Création de la Source 1 : Hautes fréquences au début
    s1 = Signal.generate_multi_freq_signal(
        duration, fs, start_time=0.2, end_time=1.0, 
        frequencies=[800, 1200, 1600], window_type='hann'
    )
    
    # Création de la Source 2 : Basses fréquences à la fin
    s2 = Signal.generate_multi_freq_signal(
        duration, fs, start_time=0.7, end_time=1.8, 
        frequencies=[300, 450, 600], window_type='hann'
    )
    
    # Regroupement en MultiSignal (Sources propres)
    sources = MultiSignal([s1, s2])
    
    # --- 2. Création du Mélange ---
    # On simule 2 micros (S=2) pour 2 sources (E=2)
    # On introduit des retards différents pour simuler la spatialisation
    # Source 1 arrive au micro 0 à t=0 et au micro 1 avec un retard de 5 samples
    # Source 2 arrive au micro 0 à t=0 et au micro 1 avec un retard de 2 samples
    delay_matrix = np.array([
        [0, 0],  # Retards vers Micro 0 (Source 1, Source 2)
        [10, 4]   # Retards vers Micro 1 (Source 1, Source 2)
    ])
    
    mixer = Mixture.create_delay_mixture(E=2, S=2, L=20, delay_matrix=delay_matrix)
    mixture_signal = mixer.apply(sources, mode='same')
    
    print("Mélange généré avec succès.")

    # --- 3. Configuration de l'algorithme de Sawada ---
    stft_params = StftParameters(
        window='hann',
        nperseg=512,
        noverlap=384, # 75% de recouvrement pour une meilleure résolution
        nfft=1024
    )
    
    # Initialisation de l'algorithme (2 sources attendues)
    bss = SawadaBSS(n_sources=2, stft_parameters=stft_params, n_iter_em=30, whitening=True)
    
    # Exécution du pipeline (STFT -> EM -> Alignement)
    print("Démarrage de la séparation (cela peut prendre quelques secondes)...")
    bss.process_signal(mixture_signal)
    
    # Récupération des signaux séparés
    separated_sources = bss.separate_source()
    
    # --- 4. Visualisation des résultats ---
    
    # Plot du mélange (ce que les micros entendent)
    fig_mix, _ = mixture_signal.plot(figsize=(12, 4))
    fig_mix.suptitle("Signaux captés par les Microphones (Mélange)")
    
    # Plot des sources séparées
    for i, src_multi in enumerate(separated_sources):
        fig, _ = src_multi.plot(overlay=False, figsize=(12, 3))
        fig.suptitle(f"Source Séparée Estimée n°{i}")
        tdoas_s0 = get_multi_tdoa_matrix(src_multi, ref_channel=0)
        print("valeur des tdoa entre le micro 0 et les micros: ", tdoas_s0)

        
    # Optionnel : Comparaison des spectrogrammes pour voir les masques
    print("Affichage des spectrogrammes de séparation...")
    for i in range(2):
        spec_i = bss.get_spectro_source_i(i)
        spec_i.plot(db=True)
        plt.gcf().suptitle(f"Spectrogramme masqué - Source {i}")

    plt.show()


def calculate_simple_cross_correlation(sig_ref: np.ndarray, sig_target: np.ndarray):
    """
    Calcule la corrélation croisée classique et les retards associés.
    """
    # Corrélation classique : on cherche la ressemblance maximale
    # mode='full' permet de tester tous les décalages possibles
    correlation = sp_signal.correlate(sig_target, sig_ref, mode='full')
    
    # Génère le vecteur des retards (lags) en nombre d'échantillons
    lags = sp_signal.correlation_lags(len(sig_target), len(sig_ref), mode='full')
    
    return correlation, lags

def estimate_tdoa_classic(signal_ref: 'Signal', signal_target: 'Signal') -> float:
    """
    Estime le TDOA entre deux objets Signal en secondes.
    """
    if signal_ref.freq != signal_target.freq:
        raise ValueError("Les fréquences d'échantillonnage doivent être identiques.")
    
    # 1. Calcul de la corrélation
    corr, lags = calculate_simple_cross_correlation(signal_ref.data, signal_target.data)
    
    # 2. Recherche du maximum (le pic de ressemblance)
    idx_max = np.argmax(np.abs(corr)) 
    delay_samples = lags[idx_max]
    
    # 3. Conversion en secondes
    return delay_samples 

def get_multi_tdoa_matrix(multi_signal: 'MultiSignal', ref_channel: int = 0):
    """
    Calcule les TDOA pour tout le MultiSignal par rapport à un canal de référence.
    """
    tdoas = {}
    for i in range(multi_signal.num_signals):
        delay = estimate_tdoa_classic(multi_signal.signals[ref_channel], multi_signal.signals[i])
        tdoas[i] = delay
    return tdoas



if __name__ == "__main__":
    test_sawada_separation()