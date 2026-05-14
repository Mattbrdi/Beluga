import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal
from scipy.io import wavfile
from ..Utils.signal_class import Signal, MultiSignal, Mixture
from ..Algo_Separation.Sawada_separation import SawadaBSS
from ..Utils.associated_dataclasses import StftParameters, EMClusteringParameters

PATH_SIGNAL_1: str = "C:/Users/BORDERIES/Desktop/Cours/Stage canada/Beluga/BSS/test_data/8296.240729000543_son_beluga.wav"
PATH_SIGNAL_2 :str = "C:/Users/BORDERIES/Desktop/Cours/Stage canada/Beluga/BSS/test_data/8295.240727103412.wav"

decay = 0 
time_lenght = 300000
START_IDX_1 : int = 185278182 +decay #Beluga
END_IDX_1 : int = START_IDX_1+time_lenght +decay
START_IDX_2: int = 40284782 #boat
END_IDX_2: int = START_IDX_2+time_lenght


def multisignal_correlation(ms_ref: MultiSignal, ms_est: MultiSignal) -> float:
    """
    Corrélation scalaire entre deux MultiSignal en concaténant tous les micros.
    On prend ensuite la valeur absolue pour gérer une éventuelle inversion de signe.
    """
    ref_data = ms_ref.data
    est_data = ms_est.data

    if ref_data.shape[0] != est_data.shape[0]:
        raise ValueError(
            f"Nombre de micros incompatible pour la corrélation : {ref_data.shape[0]} != {est_data.shape[0]}."
        )

    min_len = min(ref_data.shape[1], est_data.shape[1])
    x = ref_data[:, :min_len].ravel()
    y = est_data[:, :min_len].ravel()

    x = x - np.mean(x)
    y = y - np.mean(y)

    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return 0.0

    return float(np.dot(x, y) / denom)


def multisignal_l2_distance(ms_ref: MultiSignal, ms_est: MultiSignal) -> float:
    """
    Distance L2 entre deux MultiSignal en concaténant tous les micros.
    """
    ref_data = ms_ref.data
    est_data = ms_est.data

    if ref_data.shape[0] != est_data.shape[0]:
        raise ValueError(
            f"Nombre de micros incompatible pour la distance L2 : {ref_data.shape[0]} != {est_data.shape[0]}."
        )

    min_len = min(ref_data.shape[1], est_data.shape[1])
    x = ref_data[:, :min_len].ravel()
    y = est_data[:, :min_len].ravel()

    return float(np.linalg.norm(x - y))


def reorder_sources_by_correlation(
    reference_sources: list[MultiSignal],
    estimated_sources: list[MultiSignal]
) -> tuple[list[MultiSignal], np.ndarray]:
    """
    Réordonne les sources estimées pour maximiser la corrélation absolue
    avec les sources de référence.
    """
    n_sources = len(reference_sources)
    if len(estimated_sources) != n_sources:
        raise ValueError("Le nombre de sources de référence et estimées doit être identique.")

    correlation_matrix = np.zeros((n_sources, n_sources))
    for i, ref_source in enumerate(reference_sources):
        for j, est_source in enumerate(estimated_sources):
            correlation_matrix[i, j] = abs(multisignal_correlation(ref_source, est_source))

    if n_sources == 2:
        score_identity = correlation_matrix[0, 0] + correlation_matrix[1, 1]
        score_swap = correlation_matrix[0, 1] + correlation_matrix[1, 0]
        order = [0, 1] if score_identity >= score_swap else [1, 0]
    else:
        raise NotImplementedError("La remise dans l'ordre est seulement implémentée pour 2 sources.")

    ordered_sources = [estimated_sources[idx] for idx in order]
    return ordered_sources, correlation_matrix


def plot_multisignal_comparison(
    input_mixture: MultiSignal,
    original_sources: list[MultiSignal],
    ordered_sources: list[MultiSignal],
    stft_params: StftParameters
) -> None:
    """
    Affiche les données temporelles brutes et les spectrogrammes
    du mélange d'entrée, des sources originales et des sources séparées.
    """
    fig_mix, _,_ = input_mixture.plot_with_sounds(overlay=False, figsize=(12, 4))
    fig_mix.suptitle("Melange d'entree - signaux temporels")

    fig_mix_spec, _ = input_mixture.plot_spectrograms(
        nperseg=stft_params.nperseg,
        db=False,
        figsize=(12, 4)
    )
    fig_mix_spec.suptitle("Melange d'entree - spectrogrammes")

    for idx, (original_source, estimated_source) in enumerate(zip(original_sources, ordered_sources)):
        fig_orig, _,_ = original_source.plot_with_sounds(overlay=False, figsize=(12, 4))
        fig_orig.suptitle(f"Source originale {idx} - signaux temporels")

        fig_orig_spec, _ = original_source.plot_spectrograms(
            nperseg=stft_params.nperseg,
            figsize=(12, 4), 
            magnitude_scale = "linear",
            frequency_scale="log",
        )
        fig_orig_spec.suptitle(f"Source originale {idx} - spectrogrammes")

        fig_est, _ ,_= estimated_source.plot_with_sounds(overlay=False, figsize=(12, 4))
        fig_est.suptitle(f"Source separee {idx} - signaux temporels")

        fig_est_spec, _ = estimated_source.plot_spectrograms(
            nperseg=stft_params.nperseg,
            db=False,
            magnitude_scale='linear',
            frequency_scale='log',
            figsize=(12, 4)
        )
        fig_est_spec.suptitle(f"Source separee {idx} - spectrogrammes")

    plt.show()

def sawada_on_data():
    #Creation multisignal
    f1, donnees_1  = wavfile.read(PATH_SIGNAL_1)
    f2, donnees_2 = wavfile.read(PATH_SIGNAL_2)


    usf_data1 = donnees_1[START_IDX_1:END_IDX_1, :].T
    usf_data2 = donnees_2[START_IDX_2:END_IDX_2,:].T
    print(usf_data1.shape, usf_data2.shape)
    
    s1 = MultiSignal.from_array(data = usf_data1, fs = f1)
    s2 = MultiSignal.from_array(data = usf_data2, fs = f2)
    multi_signal  = s1 +s2
    
    #Sawada BSS
    stft_params = StftParameters(
        window='hann',
        nperseg=32768,
        noverlap=24576, # 75% de recouvrement pour une meilleure résolution
        nfft=None
    )

    em_clustering_params = EMClusteringParameters(
        n_iter= 30, 
        phi = 1.0, 
        eps = 1e-12
    )
    print("demarrage BSS")
    bss = SawadaBSS(n_sources = 2, stft_parameters= stft_params, em_clustering_parameters=em_clustering_params, whitening=True)
    
    #séparation des sources
    bss.process_signal(multi_signal= multi_signal)
    separated_sources = bss.separate_source()

    original_sources = [s1, s2]
    ordered_sources, correlation_matrix = reorder_sources_by_correlation(
        reference_sources=original_sources,
        estimated_sources=separated_sources
    )

    print("Matrice de corrélation absolue (sources originales x sources séparées) :")
    print(correlation_matrix)

    l2_distances = [
        multisignal_l2_distance(ref_source, est_source)
        for ref_source, est_source in zip(original_sources, ordered_sources)
    ]

    for idx, l2_distance in enumerate(l2_distances):
        print(f"Source {idx} - distance L2 : {l2_distance:.6f}")

    plot_multisignal_comparison(
        input_mixture=multi_signal,
        original_sources=original_sources,
        ordered_sources=ordered_sources,
        stft_params=stft_params
    )
    
    
if __name__ == "__main__":
    sawada_on_data()
