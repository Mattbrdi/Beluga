from __future__ import annotations

from collections import Counter
from math import ceil
from typing import Literal

import numpy as np


SourceCountMethod = Literal["relative_threshold", "eigengap", "explained_variance"]
AggregationMethod = Literal["median", "mode", "quantile"]


def _validate_inputs(X: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce inputs.

    Parameters
    ----------
    X:
        Complex multichannel STFT with shape ``(n_freqs, n_frames, n_channels)``.
    mask:
        Boolean time-frequency activity mask with shape ``(n_freqs, n_frames)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``X`` as a complex array and ``mask`` as a boolean array.
    """
    X_array = np.asarray(X)
    mask_array = np.asarray(mask, dtype=bool)
    if X_array.ndim != 3:
        raise ValueError(
            "X must have shape (n_freqs, n_frames, n_channels). "
            f"Got ndim={X_array.ndim}."
        )
    if mask_array.ndim != 2:
        raise ValueError(
            "mask must have shape (n_freqs, n_frames). "
            f"Got ndim={mask_array.ndim}."
        )
    if X_array.shape[:2] != mask_array.shape:
        raise ValueError(
            "X and mask have incompatible dimensions: "
            f"X.shape[:2]={X_array.shape[:2]}, mask.shape={mask_array.shape}."
        )
    if X_array.shape[2] < 1:
        raise ValueError("X must contain at least one channel.")
    return np.asarray(X_array, dtype=np.complex128), mask_array


def _compute_normalized_covariance(
    X_frequency: np.ndarray,
    mask_frequency: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, int]:
    """Compute the directional covariance for one frequency.

    Only selected frames are considered. Each selected multichannel vector
    ``x_t`` is normalized as ``y_t = x_t / ||x_t||_2``. Vectors whose norm is
    smaller than ``eps`` are ignored.

    Parameters
    ----------
    X_frequency:
        Complex STFT slice with shape ``(n_frames, n_channels)``.
    mask_frequency:
        Boolean frame mask with shape ``(n_frames,)``.
    eps:
        Numerical floor used to reject near-zero vectors.

    Returns
    -------
    tuple[np.ndarray, int]
        Hermitian covariance ``R`` with shape ``(n_channels, n_channels)`` and
        the number of nonzero selected frames actually used.
    """
    selected = np.asarray(X_frequency)[np.asarray(mask_frequency, dtype=bool)]
    n_channels = X_frequency.shape[1]
    if selected.size == 0:
        return np.zeros((n_channels, n_channels), dtype=np.complex128), 0

    norms = np.linalg.norm(selected, axis=1)
    valid = norms > eps
    if not np.any(valid):
        return np.zeros((n_channels, n_channels), dtype=np.complex128), 0

    Y = selected[valid] / norms[valid, np.newaxis]
    covariance = (Y.T @ Y.conj()) / Y.shape[0]
    covariance = 0.5 * (covariance + covariance.conj().T)
    return covariance, int(Y.shape[0])


def _eigenvalues_descending(covariance: np.ndarray) -> np.ndarray:
    """Return nonnegative eigenvalues sorted in descending order."""
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    return np.maximum(eigenvalues.real, 0.0)


def _estimate_k_relative_threshold(
    eigenvalues: np.ndarray,
    relative_threshold: float = 0.05,
    eps: float = 1e-12,
) -> float:
    """Estimate K by counting eigenvalues above ``relative_threshold * lambda_1``."""
    if eigenvalues.size == 0 or eigenvalues[0] <= eps:
        return np.nan
    threshold = float(relative_threshold) * float(eigenvalues[0])
    return float(np.sum(eigenvalues > threshold))


def _estimate_k_eigengap(
    eigenvalues: np.ndarray,
    min_eigengap_ratio: float = 3.0,
    eps: float = 1e-12,
) -> float:
    """Estimate K from the largest ratio ``lambda_i / lambda_{i+1}``.

    If the largest ratio is below ``min_eigengap_ratio``, the frequency is
    considered undecidable and ``np.nan`` is returned.
    """
    if eigenvalues.size < 2 or eigenvalues[0] <= eps:
        return np.nan
    ratios = eigenvalues[:-1] / (eigenvalues[1:] + eps)
    best_index = int(np.argmax(ratios))
    if ratios[best_index] < min_eigengap_ratio:
        return np.nan
    return float(best_index + 1)


def _estimate_k_explained_variance(
    eigenvalues: np.ndarray,
    explained_variance_threshold: float = 0.9,
    eps: float = 1e-12,
) -> float:
    """Estimate K from cumulative explained eigenvalue mass.

    This is a heuristic effective-dimension criterion. It returns the smallest
    K whose cumulative normalized eigenvalue mass exceeds the requested
    threshold.
    """
    total = float(np.sum(eigenvalues))
    if total <= eps:
        return np.nan
    threshold = float(explained_variance_threshold)
    threshold = min(max(threshold, 0.0), 1.0)
    cumulative = np.cumsum(eigenvalues / total)
    return float(int(np.searchsorted(cumulative, threshold, side="left")) + 1)


def _estimate_k_from_eigenvalues(
    eigenvalues: np.ndarray,
    method: SourceCountMethod,
    relative_threshold: float,
    min_eigengap_ratio: float,
    explained_variance_threshold: float,
    eps: float,
) -> float:
    if method == "relative_threshold":
        return _estimate_k_relative_threshold(eigenvalues, relative_threshold, eps)
    if method == "eigengap":
        return _estimate_k_eigengap(eigenvalues, min_eigengap_ratio, eps)
    if method == "explained_variance":
        return _estimate_k_explained_variance(
            eigenvalues,
            explained_variance_threshold,
            eps,
        )
    raise ValueError(f"Unknown source count method: {method!r}")


def _aggregate_source_counts(
    source_counts: np.ndarray,
    aggregation: AggregationMethod = "quantile",
    aggregation_quantile: float = 0.8,
) -> int | None:
    """Aggregate per-frequency source counts into one block-level estimate.

    Invalid frequencies must be encoded with ``np.nan`` and are ignored.

    ``median`` uses ``round``. ``mode`` chooses the smallest count in case of a
    tie. ``quantile`` uses NumPy's linear quantile followed by ``ceil`` so the
    requested high quantile remains conservative.
    """
    valid = np.asarray(source_counts, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return None

    if aggregation == "median":
        return int(round(float(np.median(valid))))
    if aggregation == "mode":
        counts = Counter(int(value) for value in valid)
        max_count = max(counts.values())
        return min(value for value, count in counts.items() if count == max_count)
    if aggregation == "quantile":
        quantile = min(max(float(aggregation_quantile), 0.0), 1.0)
        return int(ceil(float(np.quantile(valid, quantile))))
    raise ValueError(f"Unknown aggregation method: {aggregation!r}")


def estimate_num_sources(
    X: np.ndarray,
    mask: np.ndarray,
    method: SourceCountMethod = "eigengap",
    min_selected_frames: int = 20,
    relative_threshold: float = 0.05,
    min_eigengap_ratio: float = 3.0,
    explained_variance_threshold: float = 0.9,
    aggregation: AggregationMethod = "quantile",
    aggregation_quantile: float = 0.8,
    eps: float = 1e-12,
) -> dict[str, object]:
    """Estimate the approximate number of sources in a multichannel STFT block.

    Parameters
    ----------
    X:
        Complex STFT array with shape ``(n_freqs, n_frames, n_channels)``.
        Each vector ``X[f, t, :]`` is interpreted as a multichannel observation
        at frequency ``f`` and frame ``t``.
    mask:
        Boolean array with shape ``(n_freqs, n_frames)``. Only bins where
        ``mask[f, t]`` is true are considered.
    method:
        Per-frequency heuristic: ``"relative_threshold"``, ``"eigengap"``, or
        ``"explained_variance"``.
    min_selected_frames:
        Frequencies with fewer valid selected nonzero vectors are marked
        invalid and ignored in the final aggregation.
    relative_threshold:
        Used by ``"relative_threshold"``. Counts eigenvalues larger than
        ``relative_threshold * lambda_1``.
    min_eigengap_ratio:
        Used by ``"eigengap"``. If the largest eigenvalue ratio is smaller than
        this value, the frequency is marked invalid.
    explained_variance_threshold:
        Used by ``"explained_variance"``. Returns the smallest K explaining at
        least this fraction of total eigenvalue mass.
    aggregation:
        Block-level aggregation over valid frequencies: ``"median"``,
        ``"mode"``, or ``"quantile"``.
    aggregation_quantile:
        Quantile used when ``aggregation="quantile"``. The result is converted
        to an integer with ``ceil``.
    eps:
        Numerical floor for zero-norm rejection and safe divisions.

    Returns
    -------
    dict[str, object]
        Contains ``estimated_n_sources``, ``n_sources_per_frequency``,
        ``eigenvalues``, ``n_selected_frames_per_frequency``, and
        ``valid_frequencies``.

    Notes
    -----
    This is a fast heuristic, not a statistically optimal source-count
    estimator. It assumes that normalized multichannel vectors associated with
    distinct sources occupy distinct directions in ``C^M``.
    """
    X_array, mask_array = _validate_inputs(X, mask)
    n_freqs, _, n_channels = X_array.shape
    min_selected_frames = max(1, int(min_selected_frames))

    eigenvalues = np.full((n_freqs, n_channels), np.nan, dtype=float)
    source_counts = np.full(n_freqs, np.nan, dtype=float)
    selected_counts = np.zeros(n_freqs, dtype=int)
    valid_frequencies = np.zeros(n_freqs, dtype=bool)

    for frequency_index in range(n_freqs):
        covariance, n_selected = _compute_normalized_covariance(
            X_array[frequency_index],
            mask_array[frequency_index],
            eps,
        )
        selected_counts[frequency_index] = n_selected
        if n_selected < min_selected_frames:
            continue

        values = _eigenvalues_descending(covariance)
        eigenvalues[frequency_index] = values
        k_estimate = _estimate_k_from_eigenvalues(
            values,
            method,
            relative_threshold,
            min_eigengap_ratio,
            explained_variance_threshold,
            eps,
        )
        if np.isfinite(k_estimate):
            source_counts[frequency_index] = k_estimate
            valid_frequencies[frequency_index] = True

    estimated_n_sources = _aggregate_source_counts(
        source_counts,
        aggregation,
        aggregation_quantile,
    )

    return {
        "estimated_n_sources": estimated_n_sources,
        "n_sources_per_frequency": source_counts,
        "eigenvalues": eigenvalues,
        "n_selected_frames_per_frequency": selected_counts,
        "valid_frequencies": valid_frequencies,
        "method": method,
        "aggregation": aggregation,
    }
