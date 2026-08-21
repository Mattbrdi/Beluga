from __future__ import annotations

import numpy as np

from BSS.SourceCount.estimator import (
    _compute_normalized_covariance,
    _eigenvalues_descending,
    estimate_num_sources,
)


def _orthonormal_directions(n_channels: int, rng: np.random.Generator) -> np.ndarray:
    values = rng.normal(size=(n_channels, n_channels)) + 1j * rng.normal(
        size=(n_channels, n_channels)
    )
    q, _ = np.linalg.qr(values)
    return q.T


def _make_frequency_samples(
    directions: np.ndarray,
    n_frames: int,
    rng: np.random.Generator,
    noise_level: float = 0.0,
) -> np.ndarray:
    n_sources, _ = directions.shape
    labels = np.arange(n_frames) % n_sources
    rng.shuffle(labels)
    amplitudes = 0.5 + rng.random(n_frames)
    X = amplitudes[:, np.newaxis] * directions[labels]
    if noise_level > 0:
        noise = rng.normal(size=X.shape) + 1j * rng.normal(size=X.shape)
        X = X + noise_level * noise
    return X.astype(np.complex128)


def test_ideal_covariance_rank_matches_number_of_independent_directions() -> None:
    rng = np.random.default_rng(0)
    directions = _orthonormal_directions(4, rng)[:3]
    X_f = _make_frequency_samples(directions, 90, rng)
    mask_f = np.ones(90, dtype=bool)

    covariance, n_selected = _compute_normalized_covariance(X_f, mask_f)
    eigenvalues = _eigenvalues_descending(covariance)

    assert n_selected == 90
    assert np.sum(eigenvalues > 1e-10) == 3


def test_relative_threshold_counts_one_two_and_three_sources() -> None:
    rng = np.random.default_rng(1)
    directions = _orthonormal_directions(4, rng)
    X = np.stack(
        [
            _make_frequency_samples(directions[:1], 60, rng),
            _make_frequency_samples(directions[:2], 60, rng),
            _make_frequency_samples(directions[:3], 60, rng),
        ],
        axis=0,
    )
    mask = np.ones(X.shape[:2], dtype=bool)

    result = estimate_num_sources(
        X,
        mask,
        method="relative_threshold",
        relative_threshold=0.05,
        min_selected_frames=20,
        aggregation="median",
    )

    np.testing.assert_array_equal(result["n_sources_per_frequency"], [1, 2, 3])
    assert result["estimated_n_sources"] == 2


def test_eigengap_detects_two_sources_with_small_noise() -> None:
    rng = np.random.default_rng(2)
    directions = _orthonormal_directions(4, rng)
    X = _make_frequency_samples(directions[:2], 120, rng, noise_level=1e-3)[
        np.newaxis,
        :,
        :,
    ]
    mask = np.ones(X.shape[:2], dtype=bool)

    result = estimate_num_sources(
        X,
        mask,
        method="eigengap",
        min_eigengap_ratio=10.0,
        min_selected_frames=20,
    )

    assert result["n_sources_per_frequency"][0] == 2
    assert result["estimated_n_sources"] == 2


def test_explained_variance_detects_three_balanced_sources() -> None:
    rng = np.random.default_rng(3)
    directions = _orthonormal_directions(4, rng)
    X = _make_frequency_samples(directions[:3], 120, rng)[np.newaxis, :, :]
    mask = np.ones(X.shape[:2], dtype=bool)

    result = estimate_num_sources(
        X,
        mask,
        method="explained_variance",
        explained_variance_threshold=0.9,
        min_selected_frames=20,
    )

    assert result["n_sources_per_frequency"][0] == 3
    assert result["estimated_n_sources"] == 3


def test_nearly_colinear_directions_are_seen_as_one_effective_source() -> None:
    rng = np.random.default_rng(4)
    base = _orthonormal_directions(4, rng)[0]
    perturbation = _orthonormal_directions(4, rng)[1]
    near = base + 1e-3 * perturbation
    near = near / np.linalg.norm(near)
    directions = np.stack([base, near], axis=0)
    X = _make_frequency_samples(directions, 80, rng)[np.newaxis, :, :]
    mask = np.ones(X.shape[:2], dtype=bool)

    result = estimate_num_sources(
        X,
        mask,
        method="relative_threshold",
        relative_threshold=0.05,
        min_selected_frames=20,
    )

    assert result["n_sources_per_frequency"][0] == 1
    assert result["estimated_n_sources"] == 1


def test_frequency_with_too_few_observations_is_invalid() -> None:
    rng = np.random.default_rng(5)
    directions = _orthonormal_directions(4, rng)
    X = _make_frequency_samples(directions[:2], 8, rng)[np.newaxis, :, :]
    mask = np.ones(X.shape[:2], dtype=bool)

    result = estimate_num_sources(X, mask, min_selected_frames=10)

    assert np.isnan(result["n_sources_per_frequency"][0])
    assert not result["valid_frequencies"][0]
    assert result["estimated_n_sources"] is None


def test_zero_vectors_are_ignored_cleanly() -> None:
    rng = np.random.default_rng(6)
    directions = _orthonormal_directions(4, rng)
    X_f = _make_frequency_samples(directions[:2], 80, rng)
    X_f[:5] = 0.0
    X = X_f[np.newaxis, :, :]
    mask = np.ones(X.shape[:2], dtype=bool)

    result = estimate_num_sources(
        X,
        mask,
        method="relative_threshold",
        min_selected_frames=20,
    )

    assert result["n_selected_frames_per_frequency"][0] == 75
    assert result["estimated_n_sources"] == 2


def test_multiple_frequencies_are_aggregated_by_quantile() -> None:
    rng = np.random.default_rng(7)
    directions = _orthonormal_directions(4, rng)
    X = np.stack(
        [
            _make_frequency_samples(directions[:1], 80, rng),
            _make_frequency_samples(directions[:2], 80, rng),
            _make_frequency_samples(directions[:2], 80, rng),
            _make_frequency_samples(directions[:3], 80, rng),
        ],
        axis=0,
    )
    mask = np.ones(X.shape[:2], dtype=bool)

    result = estimate_num_sources(
        X,
        mask,
        method="relative_threshold",
        min_selected_frames=20,
        aggregation="quantile",
        aggregation_quantile=0.8,
    )

    np.testing.assert_array_equal(result["n_sources_per_frequency"], [1, 2, 2, 3])
    assert result["estimated_n_sources"] == 3
