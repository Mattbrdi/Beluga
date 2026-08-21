import itertools

import numpy as np

from BSS.Algo_Separation.Sawada.source_assignment import (
    assign_centroids_to_sources,
    complex_centroids_to_relative_phases,
    fit_centroid_source_trajectories,
    labels_to_source_masks,
)
from BSS.Algo_Separation.Sawada.circular_ransac import wrap_phase


def _complex_centroids_from_relative_phases(relative_phases, seed=0):
    rng = np.random.default_rng(seed)
    global_phase = rng.uniform(-np.pi, np.pi, size=relative_phases.shape[:2])
    phases = relative_phases + global_phase[:, :, None]
    return np.exp(1j * phases)


def test_complex_centroids_to_relative_phases_uses_conjugate_reference():
    relative = np.asarray(
        [
            [[0.0, 0.3, -1.2, 2.0]],
            [[0.0, -0.5, 1.1, -2.5]],
        ]
    )
    centroids = _complex_centroids_from_relative_phases(relative, seed=1)

    recovered = complex_centroids_to_relative_phases(centroids, reference_component=0)

    np.testing.assert_allclose(wrap_phase(recovered - relative), 0.0, atol=1e-12)
    np.testing.assert_allclose(recovered[:, :, 0], 0.0, atol=1e-12)

    recovered_from_fmc = complex_centroids_to_relative_phases(
        np.moveaxis(centroids, 2, 1),
        reference_component=0,
        component_axis=1,
    )
    np.testing.assert_allclose(wrap_phase(recovered_from_fmc - relative), 0.0, atol=1e-12)


def test_fit_centroid_source_trajectories_assigns_duplicate_centroids_to_same_source():
    rng = np.random.default_rng(2)
    frequencies = np.linspace(100.0, 500.0, 80)
    x = frequencies - np.mean(frequencies)
    slopes = np.asarray(
        [
            [0.0, 0.010, -0.015, 0.020],
            [0.0, -0.018, 0.012, -0.010],
        ]
    )
    intercepts = np.asarray(
        [
            [0.0, 0.4, -0.8, 1.2],
            [0.0, -1.1, 0.7, -0.3],
        ]
    )
    relative = np.empty((frequencies.size, 3, 4))
    relative[:, 0] = wrap_phase(intercepts[0][None, :] + x[:, None] * slopes[0][None, :])
    relative[:, 1] = wrap_phase(
        intercepts[0][None, :]
        + x[:, None] * slopes[0][None, :]
        + rng.normal(scale=0.01, size=(frequencies.size, 4))
    )
    relative[:, 2] = wrap_phase(intercepts[1][None, :] + x[:, None] * slopes[1][None, :])
    for frequency_index in range(frequencies.size):
        relative[frequency_index] = relative[frequency_index, rng.permutation(3)]
    centroids = _complex_centroids_from_relative_phases(relative, seed=3)

    result = fit_centroid_source_trajectories(
        centroids,
        frequencies,
        n_sources=2,
        slope_bounds=np.asarray(
            [
                [-0.001, 0.001],
                [-0.03, 0.03],
                [-0.03, 0.03],
                [-0.03, 0.03],
            ]
        ),
        residual_threshold=0.08,
        max_trials=1500,
        random_state=4,
        slope_grid_size=250,
    )

    estimated_slopes = np.asarray([model.slope_ for model in result.models])
    best_error = min(
        sum(np.linalg.norm(estimated_slopes[index] - slopes[perm[index]]) for index in range(2))
        for perm in itertools.permutations(range(2))
    )
    assert best_error < 0.01
    assert np.all(np.sum(result.labels >= 0, axis=1) >= 2)


def test_assign_centroids_to_sources_can_assign_multiple_centroids_per_frequency():
    result, frequencies, _ = fitted_two_source_assignment()

    labels, distances = assign_centroids_to_sources(
        result.relative_phases,
        frequencies,
        result.models,
        residual_threshold=0.08,
        available=np.ones_like(result.labels, dtype=bool),
    )

    assert labels.shape == result.labels.shape
    assert distances.shape == result.labels.shape
    assert np.all(np.sum(labels >= 0, axis=1) >= 2)


def test_labels_to_source_masks_merges_clusters_assigned_to_same_source():
    labels = np.asarray(
        [
            [0, 0, 1],
            [1, -1, 0],
        ]
    )
    cluster_masks = np.zeros((3, 2, 4))
    cluster_masks[0, 0] = [1, 0, 0, 0]
    cluster_masks[1, 0] = [0, 1, 0, 0]
    cluster_masks[2, 0] = [0, 0, 1, 0]
    cluster_masks[0, 1] = [1, 0, 0, 0]
    cluster_masks[2, 1] = [0, 0, 0, 1]

    source_masks = labels_to_source_masks(cluster_masks, labels, n_sources=2)

    np.testing.assert_array_equal(source_masks[0, 0], [1, 1, 0, 0])
    np.testing.assert_array_equal(source_masks[1, 0], [0, 0, 1, 0])
    np.testing.assert_array_equal(source_masks[0, 1], [0, 0, 0, 1])
    np.testing.assert_array_equal(source_masks[1, 1], [1, 0, 0, 0])


def fitted_two_source_assignment():
    rng = np.random.default_rng(5)
    frequencies = np.linspace(50.0, 350.0, 60)
    x = frequencies - np.mean(frequencies)
    slopes = np.asarray(
        [
            [0.0, 0.012, -0.010, 0.018],
            [0.0, -0.014, 0.017, -0.011],
        ]
    )
    intercepts = np.asarray(
        [
            [0.0, 0.2, -0.4, 0.9],
            [0.0, -0.9, 0.5, -0.2],
        ]
    )
    relative = np.empty((frequencies.size, 3, 4))
    relative[:, 0] = wrap_phase(intercepts[0][None, :] + x[:, None] * slopes[0][None, :])
    relative[:, 1] = wrap_phase(intercepts[0][None, :] + x[:, None] * slopes[0][None, :] + 0.01)
    relative[:, 2] = wrap_phase(intercepts[1][None, :] + x[:, None] * slopes[1][None, :])
    for frequency_index in range(frequencies.size):
        relative[frequency_index] = relative[frequency_index, rng.permutation(3)]
    centroids = _complex_centroids_from_relative_phases(relative, seed=6)
    result = fit_centroid_source_trajectories(
        centroids,
        frequencies,
        n_sources=2,
        slope_bounds=np.asarray(
            [
                [-0.001, 0.001],
                [-0.03, 0.03],
                [-0.03, 0.03],
                [-0.03, 0.03],
            ]
        ),
        residual_threshold=0.08,
        max_trials=1000,
        random_state=7,
    )
    return result, frequencies, slopes
