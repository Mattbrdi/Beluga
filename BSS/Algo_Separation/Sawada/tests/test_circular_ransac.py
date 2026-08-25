import itertools

import numpy as np

from BSS.Algo_Separation.Sawada.circular_ransac import (
    CircularLineRANSAC,
    delay_to_slope,
    sequential_centroid_circular_ransac,
    sequential_circular_ransac,
    slope_to_delay,
    wrap_phase,
)


def _match_by_slope(estimated, expected):
    remaining = list(range(len(expected)))
    pairs = []
    for model in estimated:
        distances = [np.linalg.norm(model.slope_ - expected[index]) for index in remaining]
        chosen = remaining[int(np.argmin(distances))]
        pairs.append((model, chosen))
        remaining.remove(chosen)
    return pairs


def _synthetic_line(n=80, d=3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 10.0, n)
    x = t - np.mean(t)
    intercept = rng.uniform(-np.pi, np.pi, size=d)
    slope = rng.uniform(-3.0, 3.0, size=d)
    theta = wrap_phase(intercept[None, :] + x[:, None] * slope[None, :])
    return theta, t, intercept, slope


def test_wrap_phase_range():
    values = np.asarray([-3 * np.pi, -np.pi, -0.1, 0.0, np.pi, 3 * np.pi])
    wrapped = wrap_phase(values)
    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped < np.pi)


def test_one_trajectory_without_noise_multiple_wraps():
    theta, t, intercept, slope = _synthetic_line(n=100, d=2, seed=1)
    model = CircularLineRANSAC(
        slope_bounds=(-5.0, 5.0),
        residual_threshold=0.05,
        max_trials=200,
        random_state=2,
    ).fit(theta, t)

    np.testing.assert_allclose(model.slope_, slope, atol=1e-3)
    np.testing.assert_allclose(wrap_phase(model.intercept_ - intercept), 0.0, atol=1e-3)
    assert model.n_inliers_ == theta.shape[0]


def test_circular_noise_recovers_model_approximately():
    theta, t, intercept, slope = _synthetic_line(n=120, d=3, seed=3)
    rng = np.random.default_rng(4)
    theta = wrap_phase(theta + rng.normal(scale=0.03, size=theta.shape))

    model = CircularLineRANSAC(
        slope_bounds=(-5.0, 5.0),
        residual_threshold=0.16,
        max_trials=300,
        random_state=5,
    ).fit(theta, t)

    np.testing.assert_allclose(model.slope_, slope, atol=0.05)
    np.testing.assert_allclose(wrap_phase(model.intercept_ - intercept), 0.0, atol=0.08)
    assert model.n_inliers_ > 0.85 * theta.shape[0]


def test_outliers_are_mostly_rejected():
    theta, t, _, slope = _synthetic_line(n=120, d=3, seed=6)
    rng = np.random.default_rng(7)
    n_outliers = 40
    outlier_indices = rng.choice(theta.shape[0], size=n_outliers, replace=False)
    theta[outlier_indices] = rng.uniform(-np.pi, np.pi, size=(n_outliers, theta.shape[1]))

    model = CircularLineRANSAC(
        slope_bounds=(-5.0, 5.0),
        residual_threshold=0.12,
        max_trials=1500,
        random_state=8,
    ).fit(theta, t)

    np.testing.assert_allclose(model.slope_, slope, atol=0.05)
    rejected_outliers = np.sum(~model.inlier_mask_[outlier_indices])
    assert rejected_outliers > 0.7 * n_outliers


def test_dimension_four_with_different_slopes():
    theta, t, _, slope = _synthetic_line(n=90, d=4, seed=9)
    model = CircularLineRANSAC(
        slope_bounds=(-5.0, 5.0),
        residual_threshold=0.05,
        max_trials=300,
        random_state=10,
    ).fit(theta, t)

    np.testing.assert_allclose(model.slope_, slope, atol=1e-3)


def test_crossings_at_pi_do_not_require_global_unwrapping():
    t = np.linspace(0.0, 20.0, 160)
    x = t - np.mean(t)
    intercept = np.asarray([2.9, -2.8])
    slope = np.asarray([2.4, -1.9])
    theta = wrap_phase(intercept[None, :] + x[:, None] * slope[None, :])

    model = CircularLineRANSAC(
        slope_bounds=(-4.0, 4.0),
        residual_threshold=0.05,
        max_trials=300,
        random_state=11,
    ).fit(theta, t)

    np.testing.assert_allclose(model.slope_, slope, atol=1e-3)
    np.testing.assert_allclose(wrap_phase(model.intercept_ - intercept), 0.0, atol=1e-3)


def test_modulo_ambiguity_is_resolved_by_msac():
    t = np.linspace(0.0, 8.0, 80)
    x = t - np.mean(t)
    intercept = np.asarray([0.4])
    slope = np.asarray([4.1])
    theta = wrap_phase(intercept[None, :] + x[:, None] * slope[None, :])

    model = CircularLineRANSAC(
        slope_bounds=(-8.0, 8.0),
        residual_threshold=0.04,
        max_trials=600,
        random_state=12,
        max_hypotheses_per_pair=100,
    )
    hypotheses = list(
        model._generate_hypotheses_from_pair(
            theta[0],
            theta[12],
            float(x[0]),
            float(x[12]),
            np.asarray([[-8.0, 8.0]]),
        )
    )
    assert len(hypotheses) > 1

    model.fit(theta, t)
    np.testing.assert_allclose(model.slope_, slope, atol=1e-3)


def test_large_outlier_rate():
    theta, t, _, slope = _synthetic_line(n=160, d=2, seed=13)
    rng = np.random.default_rng(14)
    outlier_indices = rng.choice(theta.shape[0], size=80, replace=False)
    theta[outlier_indices] = rng.uniform(-np.pi, np.pi, size=(80, theta.shape[1]))

    model = CircularLineRANSAC(
        slope_bounds=(-5.0, 5.0),
        residual_threshold=0.1,
        max_trials=4000,
        random_state=15,
    ).fit(theta, t)

    np.testing.assert_allclose(model.slope_, slope, atol=0.07)
    assert model.n_inliers_ >= 70


def test_centroid_mode_handles_permuted_centroids():
    rng = np.random.default_rng(16)
    f = np.linspace(100.0, 500.0, 90)
    x = f - np.mean(f)
    k = 3
    d = 3
    intercepts = rng.uniform(-np.pi, np.pi, size=(k, d))
    slopes = rng.uniform(-0.03, 0.03, size=(k, d))
    theta = np.empty((f.size, k, d))
    for source in range(k):
        theta[:, source] = wrap_phase(intercepts[source][None, :] + x[:, None] * slopes[source][None, :])
    for freq_index in range(f.size):
        theta[freq_index] = theta[freq_index, rng.permutation(k)]

    model = CircularLineRANSAC(
        slope_bounds=(-0.05, 0.05),
        residual_threshold=0.05,
        max_trials=800,
        random_state=17,
    ).fit_centroids(theta, f)

    assert model.n_inliers_ > 0.9 * f.size
    assert np.min(np.linalg.norm(slopes - model.slope_[None, :], axis=1)) < 1e-3
    assert model.selected_centroids_.shape == (f.size,)


def test_centroid_refit_uses_all_compatible_centroids():
    f = np.linspace(0.0, 4.0, 41)
    x = f - np.mean(f)
    intercept = np.asarray([0.3])
    slope = np.asarray([0.2])
    theta = np.empty((f.size, 2, 1))
    theta[:, 0] = wrap_phase(intercept[None, :] + x[:, None] * (slope[None, :] + 0.02))
    theta[:, 1] = wrap_phase(intercept[None, :] + x[:, None] * (slope[None, :] - 0.02))

    model = CircularLineRANSAC(
        slope_bounds=(-1.0, 1.0),
        residual_threshold=0.06,
        slope_grid_size=300,
        random_state=18,
    )
    refined_intercept, refined_slope = model._refit_on_candidate_centroids(
        theta,
        x,
        np.ones(f.size, dtype=bool),
        intercept,
        slope,
        np.asarray([[-1.0, 1.0]]),
        np.ones((f.size, 2), dtype=bool),
    )

    np.testing.assert_allclose(refined_slope, slope, atol=1e-3)
    np.testing.assert_allclose(wrap_phase(refined_intercept - intercept), 0.0, atol=1e-3)


def test_sequential_mode_recovers_multiple_lines_up_to_label_permutation():
    rng = np.random.default_rng(18)
    t = np.linspace(0.0, 10.0, 70)
    x = t - np.mean(t)
    slopes = np.asarray([[1.2, -0.7], [-1.4, 1.7]])
    intercepts = np.asarray([[0.2, -0.5], [1.1, 0.6]])
    theta_parts = [
        wrap_phase(intercepts[index][None, :] + x[:, None] * slopes[index][None, :])
        for index in range(2)
    ]
    theta = np.vstack(theta_parts)
    t_all = np.tile(t, 2)
    order = rng.permutation(theta.shape[0])

    result = sequential_circular_ransac(
        theta[order],
        t_all[order],
        n_components=2,
        slope_bounds=(-3.0, 3.0),
        residual_threshold=0.08,
        max_trials=1200,
        random_state=19,
    )

    assert len(result.models) == 2
    for model, source_index in _match_by_slope(result.models, slopes):
        np.testing.assert_allclose(model.slope_, slopes[source_index], atol=0.03)


def test_sequential_centroid_mode_removes_only_selected_centroids():
    rng = np.random.default_rng(20)
    f = np.linspace(80.0, 300.0, 70)
    x = f - np.mean(f)
    k = 3
    d = 2
    slopes = np.asarray([[0.015, -0.010], [-0.020, 0.018], [0.006, 0.024]])
    intercepts = rng.uniform(-np.pi, np.pi, size=(k, d))
    theta = np.empty((f.size, k, d))
    for source in range(k):
        theta[:, source] = wrap_phase(intercepts[source][None, :] + x[:, None] * slopes[source][None, :])
    for freq_index in range(f.size):
        theta[freq_index] = theta[freq_index, rng.permutation(k)]

    result = sequential_centroid_circular_ransac(
        theta,
        f,
        n_components=3,
        slope_bounds=(-0.04, 0.04),
        residual_threshold=0.06,
        max_trials=1000,
        random_state=21,
    )

    assert len(result.models) == 3
    estimated_slopes = np.asarray([model.slope_ for model in result.models])
    for permutation in itertools.permutations(range(k)):
        error = sum(
            np.linalg.norm(estimated_slopes[index] - slopes[permutation[index]])
            for index in range(k)
        )
        if error < 0.02:
            break
    else:
        raise AssertionError("Sequential centroid RANSAC did not recover the expected slopes.")
    assert np.all(np.sum(result.labels >= 0, axis=1) >= 2)


def test_delay_slope_helpers_are_inverse():
    tau = np.asarray([-0.01, 0.0, 0.02])
    np.testing.assert_allclose(slope_to_delay(delay_to_slope(tau)), tau)
