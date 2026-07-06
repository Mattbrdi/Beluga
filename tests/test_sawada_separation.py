import numpy as np

from BSS.Algo_Separation.Sawada_separation import SawadaBSS
from BSS.Utils.associated_dataclasses import (
    EMClusteringParameters,
    StftParameters,
)
from BSS.Utils.signal_class import MultiSignal


def test_sawada_returns_one_multichannel_signal_per_source():
    """Vérifie le contrat de sortie propre à SawadaBSS.

    Contrairement à ``FrequencyDomainICA.process_signal``, la méthode
    ``SawadaBSS.process_signal`` ne renvoie pas un objet résultat. Les sources
    sont récupérées ensuite avec ``separate_source()``.
    """
    rng = np.random.default_rng(12)
    np.random.seed(12)  # EMClustering utilise encore le générateur global.
    fs = 4_000
    n_samples = 4_000

    # Activité alternée : elle respecte bien l'hypothèse de parcimonie
    # temps-fréquence utilisée par les masques binaires de Sawada.
    source_0 = rng.laplace(size=n_samples)
    source_1 = rng.laplace(size=n_samples)
    source_0 *= np.tile(np.r_[np.ones(200), np.zeros(200)], 10)
    source_1 *= np.tile(np.r_[np.zeros(200), np.ones(200)], 10)

    delays = np.array([[0, 3], [2, 0], [5, 4]])
    microphones = np.zeros((3, n_samples))
    for microphone_index in range(3):
        for source_index, source in enumerate((source_0, source_1)):
            delay = int(delays[microphone_index, source_index])
            if delay == 0:
                microphones[microphone_index] += source
            else:
                microphones[microphone_index, delay:] += source[:-delay]
    microphones += 0.005 * rng.standard_normal(microphones.shape)
    mixture = MultiSignal.from_array(microphones, fs)

    model = SawadaBSS(
        n_sources=2,
        stft_parameters=StftParameters(
            nperseg=64,
            noverlap=48,
            nfft=128,
        ),
        em_clustering_parameters=EMClusteringParameters(
            n_iter=5,
            phi=1.0,
            eps=1e-12,
        ),
        whitening=True,
    )

    process_result = model.process_signal(mixture)
    separated_sources = model.separate_source()

    # Contrat Sawada : process_signal remplit le modèle et ne renvoie rien.
    assert process_result is None
    assert len(separated_sources) == model.n_sources
    assert all(
        source.num_signals == mixture.num_signals
        for source in separated_sources
    )
    assert all(source.data.shape == microphones.shape for source in separated_sources)
    assert all(np.all(np.isfinite(source.data)) for source in separated_sources)

    masks = model.get_final_masks()
    assert model.nspectro_preprocessed is not None
    n_frequencies = model.nspectro_preprocessed.Sxx.shape[1]
    n_frames = model.nspectro_preprocessed.Sxx.shape[2]
    assert masks.shape == (model.n_sources, n_frequencies, n_frames)
    assert np.all((masks == 0) | (masks == 1))
    np.testing.assert_array_equal(np.sum(masks, axis=0), np.ones((n_frequencies, n_frames)))

    # Les masques forment une partition : la somme des images séparées doit
    # donc reconstruire le mélange, indépendamment de l'ordre arbitraire ICA.
    reconstructed_mixture = sum(
        (source.data for source in separated_sources),
        start=np.zeros_like(microphones),
    )
    np.testing.assert_allclose(reconstructed_mixture, microphones, atol=1e-9)

    assert model.all_centroids.shape == (
        n_frequencies,
        mixture.num_signals,
        model.n_sources,
    )
