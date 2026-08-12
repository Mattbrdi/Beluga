from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Iterator

import numpy as np
from scipy.optimize import minimize_scalar


TWO_PI = 2.0 * np.pi


def wrap_phase(x: np.ndarray | float) -> np.ndarray | float:
    """Map angles elementwise to the interval [-pi, pi)."""
    return ((np.asarray(x) + np.pi) % TWO_PI) - np.pi


def delay_to_slope(tau: np.ndarray | float) -> np.ndarray | float:
    """Convert an acoustic delay tau to phase slope v = -2*pi*tau."""
    return -TWO_PI * np.asarray(tau)


def slope_to_delay(slope: np.ndarray | float) -> np.ndarray | float:
    """Convert a phase slope v to acoustic delay tau = -v/(2*pi)."""
    return -np.asarray(slope) / TWO_PI


@dataclass
class SequentialRansacResult:
    """Result of sequential extraction on observations shaped (N, D)."""

    models: list["CircularLineRANSAC"]
    labels: np.ndarray
    inlier_masks: list[np.ndarray]


@dataclass
class CentroidSequentialRansacResult:
    """Result of sequential extraction on centroid observations shaped (F, C, D)."""

    models: list["CircularLineRANSAC"]
    labels: np.ndarray
    available: np.ndarray


def _as_2d_theta(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 1:
        theta = theta[:, None]
    if theta.ndim != 2:
        raise ValueError("theta must have shape (N, D).")
    if theta.shape[0] < 2:
        raise ValueError("At least two observations are required.")
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta must not contain NaN or inf.")
    return wrap_phase(theta)


def _as_1d_t(t: np.ndarray, n_expected: int) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.shape[0] != n_expected:
        raise ValueError(f"t must have shape ({n_expected},).")
    if not np.all(np.isfinite(t)):
        raise ValueError("t must not contain NaN or inf.")
    return t


def _normalize_slope_bounds(slope_bounds: tuple[float, float] | np.ndarray, d: int) -> np.ndarray:
    bounds = np.asarray(slope_bounds, dtype=float)
    if bounds.shape == (2,):
        bounds = np.tile(bounds[None, :], (d, 1))
    elif bounds.shape != (d, 2):
        raise ValueError("slope_bounds must be (v_min, v_max) or an array of shape (D, 2).")
    if not np.all(np.isfinite(bounds)):
        raise ValueError("slope_bounds must be finite.")
    if not np.all(bounds[:, 0] < bounds[:, 1]):
        raise ValueError("Each slope bound must satisfy v_min < v_max.")
    return bounds


def _compute_distances(theta: np.ndarray, x: np.ndarray, intercept: np.ndarray, slope: np.ndarray) -> np.ndarray:
    pred = intercept[None, :] + x[:, None] * slope[None, :]
    residual = wrap_phase(theta - pred)
    return np.linalg.norm(residual, axis=1)


def _compute_residual_vectors(
    theta: np.ndarray,
    x: np.ndarray,
    intercept: np.ndarray,
    slope: np.ndarray,
) -> np.ndarray:
    pred = intercept[None, :] + x[:, None] * slope[None, :]
    return wrap_phase(theta - pred)


def _msac_score(distances: np.ndarray, residual_threshold: float) -> float:
    return float(np.sum(np.minimum(distances**2, residual_threshold**2)))


def _required_trials(confidence: float, inlier_ratio: float, sample_size: int = 2) -> int:
    if inlier_ratio <= 0.0:
        return math.inf  # type: ignore[return-value]
    if inlier_ratio >= 1.0:
        return 1
    success_prob = inlier_ratio**sample_size
    if success_prob <= 0.0:
        return math.inf  # type: ignore[return-value]
    if success_prob >= 1.0:
        return 1
    denominator = math.log1p(-success_prob)
    if denominator >= 0.0:
        return math.inf  # type: ignore[return-value]
    return max(1, int(math.ceil(math.log1p(-confidence) / denominator)))


def _local_maxima_indices(values: np.ndarray) -> np.ndarray:
    if values.size == 1:
        return np.asarray([0], dtype=int)

    candidates: list[int] = []
    if values[0] >= values[1]:
        candidates.append(0)
    for index in range(1, values.size - 1):
        if values[index] >= values[index - 1] and values[index] >= values[index + 1]:
            candidates.append(index)
    if values[-1] >= values[-2]:
        candidates.append(values.size - 1)
    if not candidates:
        candidates = [int(np.argmax(values))]
    return np.asarray(candidates, dtype=int)


def fit_circular_line_1d(
    phase: np.ndarray,
    x: np.ndarray,
    slope_bounds: tuple[float, float] | np.ndarray,
    slope_grid_size: int = 200,
    n_local_refinements: int = 5,
) -> tuple[float, float, float]:
    """
    Refit one circular phase line in one dimension.

    For a fixed slope v, the best intercept is

        a*(v) = arg sum_i exp(1j * (phase_i - v*x_i)).

    Therefore the slope is estimated by maximizing

        |C(v)| = |sum_i exp(1j * (phase_i - v*x_i))|

    over the physical slope interval. The objective can be multimodal because
    phases are modulo 2*pi, so the implementation first scans a grid, then
    refines the best local maxima with scipy.optimize.minimize_scalar.
    """
    phase = np.asarray(phase, dtype=float)
    x = np.asarray(x, dtype=float)
    if phase.ndim != 1 or x.ndim != 1 or phase.shape[0] != x.shape[0]:
        raise ValueError("phase and x must be one-dimensional arrays with the same length.")
    if phase.size == 0:
        raise ValueError("At least one inlier is required for refit.")
    if not np.all(np.isfinite(phase)) or not np.all(np.isfinite(x)):
        raise ValueError("phase and x must not contain NaN or inf.")

    bounds = np.asarray(slope_bounds, dtype=float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or not bounds[0] < bounds[1]:
        raise ValueError("slope_bounds must be finite and satisfy v_min < v_max.")
    if slope_grid_size < 2:
        raise ValueError("slope_grid_size must be >= 2.")
    if n_local_refinements < 1:
        raise ValueError("n_local_refinements must be >= 1.")

    v_min, v_max = float(bounds[0]), float(bounds[1])
    phase = wrap_phase(phase)

    def objective_value(slope: float) -> float:
        return float(abs(np.sum(np.exp(1j * (phase - slope * x)))))

    grid = np.linspace(v_min, v_max, int(slope_grid_size))
    values = np.asarray([objective_value(float(slope)) for slope in grid])
    maxima = _local_maxima_indices(values)
    maxima = maxima[np.argsort(values[maxima])[::-1]]
    maxima = maxima[: int(n_local_refinements)]

    best_slope = float(grid[int(maxima[0])])
    best_value = objective_value(best_slope)

    for maximum_index in maxima:
        maximum_index = int(maximum_index)
        left = float(grid[max(0, maximum_index - 1)])
        right = float(grid[min(grid.size - 1, maximum_index + 1)])
        if left == right:
            left, right = v_min, v_max

        result = minimize_scalar(
            lambda slope: -objective_value(float(slope)),
            bounds=(left, right),
            method="bounded",
        )
        candidates = [float(grid[maximum_index])]
        if result.success and np.isfinite(result.x):
            candidates.append(float(result.x))
        for candidate in candidates:
            candidate = float(np.clip(candidate, v_min, v_max))
            value = objective_value(candidate)
            if value > best_value:
                best_value = value
                best_slope = candidate

    resultant = np.sum(np.exp(1j * (phase - best_slope * x)))
    best_intercept = float(wrap_phase(np.angle(resultant)))
    return best_intercept, best_slope, best_value


class CircularLineRANSAC:
    """
    LO-MSAC estimator for linear trajectories of wrapped D-dimensional phases.

    The model is

        theta_i = intercept + slope * (t_i - mean(t)) mod 2*pi.

    Compatibility is measured with the vector circular distance

        d_i = ||wrap(theta_i - intercept - slope*x_i)||_2.

    No global phase unwrapping is performed. Hypotheses from two observations
    enumerate all modulo-compatible slopes inside physical slope_bounds, unless
    assume_no_phase_aliasing_between_sampled_points=True is selected.
    """

    def __init__(
        self,
        slope_bounds: tuple[float, float] | np.ndarray,
        residual_threshold: float,
        max_trials: int = 1000,
        confidence: float = 0.99,
        min_inliers: int = 2,
        min_dx: float = 1e-12,
        local_optimization_steps: int = 3,
        slope_grid_size: int = 200,
        n_local_refinements: int = 5,
        max_hypotheses_per_pair: int = 1000,
        assume_no_phase_aliasing_between_sampled_points: bool = False,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        if residual_threshold <= 0.0 or not np.isfinite(residual_threshold):
            raise ValueError("residual_threshold must be a positive finite value.")
        if max_trials < 1:
            raise ValueError("max_trials must be >= 1.")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must satisfy 0 < confidence < 1.")
        if min_inliers < 2:
            raise ValueError("min_inliers must be >= 2.")
        if min_dx < 0.0 or not np.isfinite(min_dx):
            raise ValueError("min_dx must be non-negative and finite.")
        if local_optimization_steps < 0:
            raise ValueError("local_optimization_steps must be >= 0.")
        if max_hypotheses_per_pair < 1:
            raise ValueError("max_hypotheses_per_pair must be >= 1.")

        self.slope_bounds = slope_bounds
        self.residual_threshold = float(residual_threshold)
        self.max_trials = int(max_trials)
        self.confidence = float(confidence)
        self.min_inliers = int(min_inliers)
        self.min_dx = float(min_dx)
        self.local_optimization_steps = int(local_optimization_steps)
        self.slope_grid_size = int(slope_grid_size)
        self.n_local_refinements = int(n_local_refinements)
        self.max_hypotheses_per_pair = int(max_hypotheses_per_pair)
        self.assume_no_phase_aliasing_between_sampled_points = bool(
            assume_no_phase_aliasing_between_sampled_points
        )
        self.random_state = random_state

        self.intercept_: np.ndarray | None = None
        self.slope_: np.ndarray | None = None
        self.inlier_mask_: np.ndarray | None = None
        self.score_: float | None = None
        self.n_inliers_: int = 0
        self.n_trials_: int = 0
        self.t_ref_: float | None = None
        self.converged_: bool = False
        self.selected_centroids_: np.ndarray | None = None
        self.frequency_inliers_: np.ndarray | None = None

    def _rng(self) -> np.random.Generator:
        if isinstance(self.random_state, np.random.Generator):
            return self.random_state
        return np.random.default_rng(self.random_state)

    def _generate_hypotheses_from_pair(
        self,
        theta1: np.ndarray,
        theta2: np.ndarray,
        x1: float,
        x2: float,
        slope_bounds: np.ndarray,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        dx = float(x2 - x1)
        if abs(dx) < self.min_dx:
            return

        dtheta = theta2 - theta1
        if self.assume_no_phase_aliasing_between_sampled_points:
            slope = wrap_phase(dtheta) / dx
            if np.all((slope >= slope_bounds[:, 0]) & (slope <= slope_bounds[:, 1])):
                intercept = wrap_phase(theta1 - slope * x1)
                yield np.asarray(intercept, dtype=float), np.asarray(slope, dtype=float)
            return

        candidate_slopes_per_dim: list[list[float]] = []
        hypothesis_count = 1
        for dim in range(theta1.size):
            v_min, v_max = slope_bounds[dim]
            lo = min(v_min * dx, v_max * dx)
            hi = max(v_min * dx, v_max * dx)
            n_min = math.ceil((lo - dtheta[dim]) / TWO_PI)
            n_max = math.floor((hi - dtheta[dim]) / TWO_PI)
            if n_max < n_min:
                return

            candidates = [
                float((dtheta[dim] + TWO_PI * n) / dx)
                for n in range(n_min, n_max + 1)
            ]
            if not candidates:
                return
            candidate_slopes_per_dim.append(candidates)
            hypothesis_count *= len(candidates)
            if hypothesis_count > self.max_hypotheses_per_pair:
                return

        for slope_tuple in product(*candidate_slopes_per_dim):
            slope = np.asarray(slope_tuple, dtype=float)
            intercept = wrap_phase(theta1 - slope * x1)
            yield np.asarray(intercept, dtype=float), slope

    def _evaluate_model(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        intercept: np.ndarray,
        slope: np.ndarray,
    ) -> tuple[float, np.ndarray, int, np.ndarray]:
        distances = _compute_distances(theta, x, intercept, slope)
        inliers = distances < self.residual_threshold
        n_inliers = int(np.sum(inliers))
        score = _msac_score(distances, self.residual_threshold)
        return score, inliers, n_inliers, distances

    def _refit_on_inliers(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        inliers: np.ndarray,
        slope_bounds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if int(np.sum(inliers)) < self.min_inliers:
            raise ValueError("Not enough inliers for refit.")
        theta_i = theta[inliers]
        x_i = x[inliers]
        d = theta.shape[1]
        intercept = np.empty(d, dtype=float)
        slope = np.empty(d, dtype=float)
        for dim in range(d):
            intercept[dim], slope[dim], _ = fit_circular_line_1d(
                theta_i[:, dim],
                x_i,
                slope_bounds[dim],
                slope_grid_size=self.slope_grid_size,
                n_local_refinements=self.n_local_refinements,
            )
        return wrap_phase(intercept), slope

    def _local_optimize(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        intercept: np.ndarray,
        slope: np.ndarray,
        inliers: np.ndarray,
        score: float,
        slope_bounds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        best_intercept = intercept
        best_slope = slope
        best_inliers = inliers
        best_score = score

        current_inliers = inliers.copy()
        for _ in range(self.local_optimization_steps):
            if int(np.sum(current_inliers)) < self.min_inliers:
                break
            try:
                refined_intercept, refined_slope = self._refit_on_inliers(
                    theta,
                    x,
                    current_inliers,
                    slope_bounds,
                )
            except ValueError:
                break
            refined_score, refined_inliers, refined_n_inliers, _ = self._evaluate_model(
                theta,
                x,
                refined_intercept,
                refined_slope,
            )
            if refined_n_inliers < self.min_inliers:
                break
            if refined_score < best_score:
                best_intercept = refined_intercept
                best_slope = refined_slope
                best_inliers = refined_inliers
                best_score = refined_score
            if np.array_equal(refined_inliers, current_inliers):
                break
            current_inliers = refined_inliers

        return best_intercept, best_slope, best_inliers, best_score

    def fit(self, theta: np.ndarray, t: np.ndarray) -> "CircularLineRANSAC":
        theta = _as_2d_theta(theta)
        t = _as_1d_t(t, theta.shape[0])
        n, d = theta.shape
        slope_bounds = _normalize_slope_bounds(self.slope_bounds, d)

        self.t_ref_ = float(np.mean(t))
        x = t - self.t_ref_
        rng = self._rng()

        best_score = math.inf
        best_intercept: np.ndarray | None = None
        best_slope: np.ndarray | None = None
        best_inliers: np.ndarray | None = None
        adaptive_max_trials = self.max_trials
        self.n_trials_ = 0

        while self.n_trials_ < adaptive_max_trials:
            self.n_trials_ += 1
            sample = rng.choice(n, size=2, replace=False)
            i1, i2 = int(sample[0]), int(sample[1])

            for intercept, slope in self._generate_hypotheses_from_pair(
                theta[i1],
                theta[i2],
                float(x[i1]),
                float(x[i2]),
                slope_bounds,
            ):
                score, inliers, n_inliers, _ = self._evaluate_model(theta, x, intercept, slope)
                if n_inliers < self.min_inliers:
                    continue

                intercept, slope, inliers, score = self._local_optimize(
                    theta,
                    x,
                    intercept,
                    slope,
                    inliers,
                    score,
                    slope_bounds,
                )
                n_inliers = int(np.sum(inliers))
                if n_inliers < self.min_inliers:
                    continue

                if score < best_score:
                    best_score = score
                    best_intercept = intercept
                    best_slope = slope
                    best_inliers = inliers
                    required = _required_trials(self.confidence, n_inliers / n, sample_size=2)
                    adaptive_max_trials = min(self.max_trials, adaptive_max_trials, required)

        if best_intercept is None or best_slope is None or best_inliers is None:
            raise RuntimeError("RANSAC did not find a valid circular line model.")

        self.intercept_ = best_intercept
        self.slope_ = best_slope
        self.inlier_mask_ = best_inliers
        self.score_ = float(best_score)
        self.n_inliers_ = int(np.sum(best_inliers))
        self.converged_ = self.n_trials_ < self.max_trials
        self.selected_centroids_ = None
        self.frequency_inliers_ = None
        return self

    def _check_is_fitted(self) -> tuple[np.ndarray, np.ndarray, float]:
        if self.intercept_ is None or self.slope_ is None or self.t_ref_ is None:
            raise RuntimeError("The model is not fitted.")
        return self.intercept_, self.slope_, self.t_ref_

    def predict_phase(self, t: np.ndarray) -> np.ndarray:
        intercept, slope, t_ref = self._check_is_fitted()
        t = np.asarray(t, dtype=float)
        if t.ndim != 1:
            raise ValueError("t must be one-dimensional.")
        x = t - t_ref
        return wrap_phase(intercept[None, :] + x[:, None] * slope[None, :])

    def residuals(self, theta: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Return scalar circular distances d_i for each observation."""
        intercept, slope, t_ref = self._check_is_fitted()
        theta = _as_2d_theta(theta)
        t = _as_1d_t(t, theta.shape[0])
        return _compute_distances(theta, t - t_ref, intercept, slope)

    def residual_vectors(self, theta: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Return wrapped vector residuals with shape (N, D)."""
        intercept, slope, t_ref = self._check_is_fitted()
        theta = _as_2d_theta(theta)
        t = _as_1d_t(t, theta.shape[0])
        return _compute_residual_vectors(theta, t - t_ref, intercept, slope)

    def score(self, theta: np.ndarray, t: np.ndarray) -> float:
        distances = self.residuals(theta, t)
        return _msac_score(distances, self.residual_threshold)

    def _centroid_distances(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        intercept: np.ndarray,
        slope: np.ndarray,
        available: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred = intercept[None, None, :] + x[:, None, None] * slope[None, None, :]
        distances_fc = np.linalg.norm(wrap_phase(theta - pred), axis=2)
        distances_fc = np.where(available, distances_fc, np.inf)
        selected = np.argmin(distances_fc, axis=1)
        distances = distances_fc[np.arange(theta.shape[0]), selected]
        selected = np.where(np.isfinite(distances), selected, -1)
        return distances, selected.astype(int), distances_fc

    def _evaluate_centroid_model(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        intercept: np.ndarray,
        slope: np.ndarray,
        available: np.ndarray,
    ) -> tuple[float, np.ndarray, int, np.ndarray, np.ndarray]:
        distances, selected, _ = self._centroid_distances(theta, x, intercept, slope, available)
        inliers = distances < self.residual_threshold
        n_inliers = int(np.sum(inliers))
        score = _msac_score(distances, self.residual_threshold)
        return score, inliers, n_inliers, selected, distances

    def _refit_on_candidate_centroids(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        inliers: np.ndarray,
        intercept: np.ndarray,
        slope: np.ndarray,
        slope_bounds: np.ndarray,
        available: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        _, _, distances_fc = self._centroid_distances(theta, x, intercept, slope, available)
        candidate_mask = (
            inliers[:, None]
            & available
            & (distances_fc < self.residual_threshold)
        )
        frequency_indices, centroid_indices = np.nonzero(candidate_mask)
        if frequency_indices.size < self.min_inliers:
            raise ValueError("Not enough candidate centroid inliers for refit.")

        candidate_theta = theta[frequency_indices, centroid_indices]
        candidate_x = x[frequency_indices]
        return self._refit_on_inliers(
            candidate_theta,
            candidate_x,
            np.ones(candidate_theta.shape[0], dtype=bool),
            slope_bounds,
        )

    def _local_optimize_centroids(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        intercept: np.ndarray,
        slope: np.ndarray,
        inliers: np.ndarray,
        selected: np.ndarray,
        score: float,
        slope_bounds: np.ndarray,
        available: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        best_intercept = intercept
        best_slope = slope
        best_inliers = inliers
        best_selected = selected
        best_score = score

        current_inliers = inliers.copy()
        current_selected = selected.copy()
        current_intercept = intercept
        current_slope = slope
        for _ in range(self.local_optimization_steps):
            if int(np.sum(current_inliers & (current_selected >= 0))) < self.min_inliers:
                break
            try:
                refined_intercept, refined_slope = self._refit_on_candidate_centroids(
                    theta,
                    x,
                    current_inliers,
                    current_intercept,
                    current_slope,
                    slope_bounds,
                    available,
                )
            except ValueError:
                break
            refined_score, refined_inliers, refined_n, refined_selected, _ = (
                self._evaluate_centroid_model(theta, x, refined_intercept, refined_slope, available)
            )
            if refined_n < self.min_inliers:
                break
            if refined_score < best_score:
                best_intercept = refined_intercept
                best_slope = refined_slope
                best_inliers = refined_inliers
                best_selected = refined_selected
                best_score = refined_score
            if np.array_equal(refined_inliers, current_inliers) and np.array_equal(
                refined_selected,
                current_selected,
            ):
                break
            current_inliers = refined_inliers
            current_selected = refined_selected
            current_intercept = refined_intercept
            current_slope = refined_slope

        return best_intercept, best_slope, best_inliers, best_selected, best_score

    def fit_centroids(
        self,
        theta: np.ndarray,
        frequencies: np.ndarray,
        available: np.ndarray | None = None,
    ) -> "CircularLineRANSAC":
        """
        Fit one trajectory to centroid observations shaped (F, C, D).

        For scoring, each frequency f contributes only its closest available centroid:

            d_f = min_c ||wrap(theta_fc - intercept - slope*(freq_f - mean(freq)))||_2.

        During local refinement, however, every available centroid compatible
        with the current trajectory, i.e. below residual_threshold on an inlier
        frequency, is used. This avoids assuming that there is only one centroid
        per physical source at each frequency.
        """
        theta = np.asarray(theta, dtype=float)
        if theta.ndim != 3:
            raise ValueError("theta must have shape (F, C, D).")
        if theta.shape[0] < 2 or theta.shape[1] < 1:
            raise ValueError("At least two frequencies and one centroid are required.")
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta must not contain NaN or inf.")
        theta = wrap_phase(theta)
        f_count, c_count, d = theta.shape
        frequencies = _as_1d_t(frequencies, f_count)
        slope_bounds = _normalize_slope_bounds(self.slope_bounds, d)
        if available is None:
            available = np.ones((f_count, c_count), dtype=bool)
        else:
            available = np.asarray(available, dtype=bool)
            if available.shape != (f_count, c_count):
                raise ValueError("available must have shape (F, C).")

        if int(np.sum(np.any(available, axis=1))) < 2:
            raise RuntimeError("Not enough frequencies with available centroids.")

        self.t_ref_ = float(np.mean(frequencies))
        x = frequencies - self.t_ref_
        rng = self._rng()

        best_score = math.inf
        best_intercept: np.ndarray | None = None
        best_slope: np.ndarray | None = None
        best_inliers: np.ndarray | None = None
        best_selected: np.ndarray | None = None
        adaptive_max_trials = self.max_trials
        self.n_trials_ = 0

        available_frequencies = np.flatnonzero(np.any(available, axis=1))
        while self.n_trials_ < adaptive_max_trials:
            self.n_trials_ += 1
            f1, f2 = rng.choice(available_frequencies, size=2, replace=False)
            c1 = int(rng.choice(np.flatnonzero(available[f1])))
            c2 = int(rng.choice(np.flatnonzero(available[f2])))

            for intercept, slope in self._generate_hypotheses_from_pair(
                theta[f1, c1],
                theta[f2, c2],
                float(x[f1]),
                float(x[f2]),
                slope_bounds,
            ):
                score, inliers, n_inliers, selected, _ = self._evaluate_centroid_model(
                    theta,
                    x,
                    intercept,
                    slope,
                    available,
                )
                if n_inliers < self.min_inliers:
                    continue

                intercept, slope, inliers, selected, score = self._local_optimize_centroids(
                    theta,
                    x,
                    intercept,
                    slope,
                    inliers,
                    selected,
                    score,
                    slope_bounds,
                    available,
                )
                n_inliers = int(np.sum(inliers))
                if n_inliers < self.min_inliers:
                    continue

                if score < best_score:
                    best_score = score
                    best_intercept = intercept
                    best_slope = slope
                    best_inliers = inliers
                    best_selected = selected
                    required = _required_trials(self.confidence, n_inliers / f_count, sample_size=2)
                    adaptive_max_trials = min(self.max_trials, adaptive_max_trials, required)

        if (
            best_intercept is None
            or best_slope is None
            or best_inliers is None
            or best_selected is None
        ):
            raise RuntimeError("RANSAC did not find a valid centroid trajectory.")

        self.intercept_ = best_intercept
        self.slope_ = best_slope
        self.inlier_mask_ = best_inliers
        self.frequency_inliers_ = best_inliers
        self.selected_centroids_ = best_selected
        self.score_ = float(best_score)
        self.n_inliers_ = int(np.sum(best_inliers))
        self.converged_ = self.n_trials_ < self.max_trials
        return self


def sequential_circular_ransac(
    theta: np.ndarray,
    t: np.ndarray,
    n_components: int,
    *,
    global_assignment: bool = True,
    mark_outliers: bool = True,
    **ransac_kwargs: object,
) -> SequentialRansacResult:
    """Extract several circular trajectories by sequential RANSAC."""
    theta_all = _as_2d_theta(theta)
    t_all = _as_1d_t(t, theta_all.shape[0])
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")

    base_seed = ransac_kwargs.pop("random_state", None)
    rng = np.random.default_rng(base_seed)
    remaining = np.ones(theta_all.shape[0], dtype=bool)
    labels = np.full(theta_all.shape[0], -1, dtype=int)
    models: list[CircularLineRANSAC] = []
    inlier_masks: list[np.ndarray] = []

    for component in range(n_components):
        remaining_indices = np.flatnonzero(remaining)
        if remaining_indices.size < 2:
            break
        model = CircularLineRANSAC(
            random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
            **ransac_kwargs,
        )
        try:
            model.fit(theta_all[remaining_indices], t_all[remaining_indices])
        except RuntimeError:
            break
        local_inliers = np.asarray(model.inlier_mask_, dtype=bool)
        global_inliers = np.zeros(theta_all.shape[0], dtype=bool)
        global_inliers[remaining_indices] = local_inliers
        if not np.any(global_inliers):
            break
        labels[global_inliers] = component
        remaining[global_inliers] = False
        models.append(model)
        inlier_masks.append(global_inliers)

    if global_assignment and models:
        distances = np.vstack([model.residuals(theta_all, t_all) for model in models])
        best_model = np.argmin(distances, axis=0)
        best_distance = distances[best_model, np.arange(theta_all.shape[0])]
        threshold = models[0].residual_threshold
        labels = best_model.astype(int)
        if mark_outliers:
            labels[best_distance >= threshold] = -1

    return SequentialRansacResult(models=models, labels=labels, inlier_masks=inlier_masks)


def sequential_centroid_circular_ransac(
    theta: np.ndarray,
    frequencies: np.ndarray,
    n_components: int,
    available: np.ndarray | None = None,
    **ransac_kwargs: object,
) -> CentroidSequentialRansacResult:
    """Extract several trajectories from centroid observations shaped (F, C, D)."""
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 3:
        raise ValueError("theta must have shape (F, C, D).")
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")
    frequencies = _as_1d_t(frequencies, theta.shape[0])

    base_seed = ransac_kwargs.pop("random_state", None)
    rng = np.random.default_rng(base_seed)
    f_count, c_count, _ = theta.shape
    if available is None:
        available = np.ones((f_count, c_count), dtype=bool)
    else:
        available = np.asarray(available, dtype=bool).copy()
        if available.shape != (f_count, c_count):
            raise ValueError("available must have shape (F, C).")
    labels = np.full((f_count, c_count), -1, dtype=int)
    models: list[CircularLineRANSAC] = []

    for component in range(n_components):
        if int(np.sum(np.any(available, axis=1))) < 2:
            break
        model = CircularLineRANSAC(
            random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
            **ransac_kwargs,
        )
        try:
            model.fit_centroids(theta, frequencies, available=available)
        except RuntimeError:
            break

        selected = np.asarray(model.selected_centroids_, dtype=int)
        inliers = np.asarray(model.frequency_inliers_, dtype=bool)
        valid = inliers & (selected >= 0)
        if not np.any(valid):
            break
        rows = np.flatnonzero(valid)
        cols = selected[valid]
        labels[rows, cols] = component
        available[rows, cols] = False
        models.append(model)

    return CentroidSequentialRansacResult(models=models, labels=labels, available=available)
