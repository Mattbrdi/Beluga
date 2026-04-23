import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from Beluga.BSS.signal_class import Signal, MultiSignal, Mixture
from Beluga.BSS.Sawada_separation import SawadaBSS, StftParameters

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
        
    # Optionnel : Comparaison des spectrogrammes pour voir les masques
    print("Affichage des spectrogrammes de séparation...")
    for i in range(2):
        spec_i = bss.get_spectro_source_i(i)
        spec_i.plot(db=True)
        plt.gcf().suptitle(f"Spectrogramme masqué - Source {i}")

    plt.show()

if __name__ == "__main__":
    test_sawada_separation()