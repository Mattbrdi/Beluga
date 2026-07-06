import numpy as np

from BSS.Algo_Separation.Frequency_ica_separation import FrequencyDomainICA
from BSS.Utils.associated_dataclasses import StftParameters
from BSS.Utils.signal_class import MultiSignal


def test_phase_regression_recovers_fractional_tdoas():
    fs = 8_000
    frequencies = np.linspace(0, fs / 2, 513)
    expected_samples = np.array(
        [[0.0, 3.25, -2.0, 1.0], [0.0, -4.0, 2.5, 5.0]]
    )
    expected_seconds = expected_samples / fs
    mixing = np.empty((len(frequencies), 4, 2), dtype=complex)
    for frequency_index, frequency in enumerate(frequencies):
        mixing[frequency_index] = np.exp(
            -2j * np.pi * frequency * expected_seconds.T
        )

    model = FrequencyDomainICA(n_sources=2, max_tdoa_seconds=0.002)
    model.mixing_matrices = mixing
    model.bin_reliability = np.ones(len(frequencies))
    model.input_spectrogram = type("Spectrum", (), {"f": frequencies})()

    estimated_samples = model.estimate_tdoas() * fs
    np.testing.assert_allclose(estimated_samples, expected_samples, atol=0.03)


def test_phase_regression_rejects_misassigned_frequency_bins():
    """Des bins d'une autre source ne doivent pas déplacer le TDOA estimé."""
    fs = 8_000
    frequencies = np.linspace(0, fs / 2, 513)
    expected_delay = -3.0 / fs
    phase = np.angle(np.exp(-2j * np.pi * frequencies * expected_delay))

    # Simule un changement de permutation ICA sur un tiers des fréquences.
    corrupted = np.arange(len(frequencies)) % 3 == 0
    phase[corrupted] = 0.0

    model = FrequencyDomainICA(n_sources=2, max_tdoa_seconds=0.002)
    estimated_delay = model._delay_from_phase(
        frequencies, phase, np.ones_like(frequencies)
    )

    np.testing.assert_allclose(estimated_delay * fs, -3.0, atol=0.03)


def test_process_signal_returns_sources_and_tdoas_with_expected_shapes():
    rng = np.random.default_rng(4)
    fs = 4_000
    n_samples = 8_000

    # Deux sources non gaussiennes, modulées par des enveloppes différentes.
    source_0 = rng.laplace(size=n_samples)
    source_1 = rng.laplace(size=n_samples)
    source_0 *= np.tile(np.r_[np.ones(400), np.zeros(400)], 10)
    source_1 *= np.tile(np.r_[np.zeros(400), np.ones(400)], 10)
    delays = np.array([[0, 3], [2, 0], [5, 4]])

    microphones = np.zeros((3, n_samples))
    for microphone in range(3):
        for source_index, source in enumerate((source_0, source_1)):
            delay = int(delays[microphone, source_index])
            if delay == 0:
                microphones[microphone] += source
            else:
                microphones[microphone, delay:] += source[:-delay]
    microphones += 0.005 * rng.standard_normal(microphones.shape)

    model = FrequencyDomainICA(
        n_sources=2,
        stft_parameters=StftParameters(nperseg=128, noverlap=96, nfft=256),
        n_iter=40,
        max_tdoa_seconds=0.005,
        random_state=2,
    )
    result = model.process_signal(MultiSignal.from_array(microphones, fs))

    assert len(result.sources) == 2
    assert all(source.data.shape == microphones.shape for source in result.sources)
    assert result.tdoas.shape == (2, 3)
    assert np.all(np.isfinite(result.tdoas))
    np.testing.assert_array_equal(result.tdoas[:, 0], 0.0)
    np.testing.assert_allclose(result.tdoas_samples, result.tdoas * fs)

    # L'ordre ICA est arbitraire : on retient la meilleure des deux permutations.
    expected_samples = (delays - delays[0]).T
    identity_error = np.mean((result.tdoas_samples - expected_samples) ** 2)
    swapped_error = np.mean((result.tdoas_samples[::-1] - expected_samples) ** 2)
    aligned = result.tdoas_samples if identity_error <= swapped_error else result.tdoas_samples[::-1]
    np.testing.assert_allclose(aligned, expected_samples, atol=0.5)
