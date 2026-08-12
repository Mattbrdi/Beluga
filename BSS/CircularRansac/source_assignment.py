from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circular_ransac import (
    CircularLineRANSAC,
    sequential_centroid_circular_ransac,
    wrap_phase,
)


@dataclass
class CentroidSourceAssignment:
    """Result of assigning complex frequency-wise centroids to source trajectories."""

    models: list[CircularLineRANSAC]
    labels: np.ndarray
    distances: np.ndarray
    relative_phases: np.ndarray
    selected_labels: np.ndarray
    available: np.ndarray


def complex_centroids_to_relative_phases(
    centroids: np.ndarray,
    reference_component: int = 0,
    component_axis: int = -1,
) -> np.ndarray:
    """
    Convert complex centroids to relative phase vectors.

    Sawada centroids live in C^M. For delay-like acoustic trajectories, the
    physically meaningful quantities are phase differences between microphones,
    not the absolute complex phase. This function removes the global phase by
    multiplying every component by the conjugate of one reference component:

        z_rel[f, c, m] = z[f, c, m] * conj(z[f, c, ref]).

    The returned observation is

        theta[f, c, m] = arg(z_rel[f, c, m]) in [-pi, pi).

    This is intentionally not a division by the reference component. The phase
    effect is the same, but multiplication avoids exploding magnitudes when the
    reference component has a small modulus.
    """
    centroids = np.asarray(centroids)
    if centroids.ndim != 3:
        raise ValueError("centroids must have three axes: frequency, centroid, component.")
    if not np.iscomplexobj(centroids):
        raise ValueError("centroids must be complex-valued.")
    if not np.all(np.isfinite(centroids.real)) or not np.all(np.isfinite(centroids.imag)):
        raise ValueError("centroids must not contain NaN or inf.")
    component_axis = int(component_axis)
    if component_axis < 0:
        component_axis += centroids.ndim
    if component_axis not in {1, 2}:
        raise ValueError("component_axis must be 1 or 2 for a three-dimensional centroid tensor.")
    if component_axis != 2:
        centroids = np.moveaxis(centroids, component_axis, 2)

    n_components = centroids.shape[2]
    if not 0 <= reference_component < n_components:
        raise ValueError("reference_component is out of bounds.")

    reference = centroids[:, :, reference_component:reference_component + 1]
    relative = centroids * np.conj(reference)
    phases = wrap_phase(np.angle(relative))
    phases[:, :, reference_component] = 0.0
    return np.asarray(phases, dtype=float)


def distances_to_source_trajectories(
    relative_phases: np.ndarray,
    frequencies: np.ndarray,
    models: list[CircularLineRANSAC],
    available: np.ndarray | None = None,
) -> np.ndarray:
    """Return distances with shape (K, F, C) from every centroid to every model."""
    relative_phases = np.asarray(relative_phases, dtype=float)
    if relative_phases.ndim != 3:
        raise ValueError("relative_phases must have shape (F, C, D).")
    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.ndim != 1 or frequencies.shape[0] != relative_phases.shape[0]:
        raise ValueError("frequencies must have shape (F,).")
    if not models:
        raise ValueError("At least one fitted model is required.")

    if available is None:
        available = np.ones(relative_phases.shape[:2], dtype=bool)
    else:
        available = np.asarray(available, dtype=bool)
        if available.shape != relative_phases.shape[:2]:
            raise ValueError("available must have shape (F, C).")

    distances = np.empty((len(models), relative_phases.shape[0], relative_phases.shape[1]))
    for model_index, model in enumerate(models):
        if model.intercept_ is None or model.slope_ is None or model.t_ref_ is None:
            raise RuntimeError("All models must be fitted before assignment.")
        x = frequencies - model.t_ref_
        prediction = model.intercept_[None, None, :] + x[:, None, None] * model.slope_[None, None, :]
        distances[model_index] = np.linalg.norm(wrap_phase(relative_phases - prediction), axis=2)

    distances[:, ~available] = np.inf
    return distances


def assign_centroids_to_sources(
    relative_phases: np.ndarray,
    frequencies: np.ndarray,
    models: list[CircularLineRANSAC],
    residual_threshold: float | None = None,
    available: np.ndarray | None = None,
    mark_outliers: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign each available centroid to the nearest source trajectory.

    This is a global assignment step after the RANSAC trajectories have been
    estimated. Unlike scoring in fit_centroids, it assigns every centroid, not
    only the closest centroid of each frequency. Therefore, if two centroids at
    the same frequency are compatible with the same source trajectory, both can
    receive that source label.
    """
    distances = distances_to_source_trajectories(
        relative_phases,
        frequencies,
        models,
        available=available,
    )
    best_label = np.argmin(distances, axis=0).astype(int)
    best_distance = np.take_along_axis(distances, best_label[None, :, :], axis=0)[0]

    labels = best_label
    if mark_outliers:
        if residual_threshold is None:
            residual_threshold = models[0].residual_threshold
        labels = labels.copy()
        labels[best_distance >= float(residual_threshold)] = -1
    labels[~np.isfinite(best_distance)] = -1
    return labels, best_distance


def fit_centroid_source_trajectories(
    complex_centroids: np.ndarray,
    frequencies: np.ndarray,
    n_sources: int,
    *,
    slope_bounds: tuple[float, float] | np.ndarray,
    residual_threshold: float,
    reference_component: int = 0,
    component_axis: int = -1,
    available: np.ndarray | None = None,
    assignment_threshold: float | None = None,
    mark_outliers: bool = True,
    **ransac_kwargs: object,
) -> CentroidSourceAssignment:
    """
    Fit source phase trajectories from complex centroids and assign all centroids.

    The pipeline is:

    1. convert complex centroids z[f, c, m] to relative phases using
       z * conj(z_ref);
    2. run sequential circular RANSAC in centroid mode;
    3. assign every available centroid to its nearest fitted trajectory.

    The RANSAC score still uses one closest centroid per frequency. The final
    assignment is denser: all centroids compatible with a trajectory can be
    assigned to that source.
    """
    if n_sources < 1:
        raise ValueError("n_sources must be >= 1.")
    relative_phases = complex_centroids_to_relative_phases(
        complex_centroids,
        reference_component=reference_component,
        component_axis=component_axis,
    )
    if available is not None:
        available = np.asarray(available, dtype=bool)
        if available.shape != relative_phases.shape[:2]:
            raise ValueError("available must have shape (F, C).")
        initial_available = available.copy()
    else:
        initial_available = np.ones(relative_phases.shape[:2], dtype=bool)

    sequential = sequential_centroid_circular_ransac(
        relative_phases,
        frequencies,
        n_components=n_sources,
        available=initial_available,
        slope_bounds=slope_bounds,
        residual_threshold=residual_threshold,
        random_state=ransac_kwargs.pop("random_state", None),
        **ransac_kwargs,
    )
    labels, distances = assign_centroids_to_sources(
        relative_phases,
        frequencies,
        sequential.models,
        residual_threshold=assignment_threshold
        if assignment_threshold is not None
        else residual_threshold,
        available=initial_available,
        mark_outliers=mark_outliers,
    )
    return CentroidSourceAssignment(
        models=sequential.models,
        labels=labels,
        distances=distances,
        relative_phases=relative_phases,
        selected_labels=sequential.labels,
        available=initial_available,
    )


def labels_to_source_masks(
    cluster_masks: np.ndarray,
    labels: np.ndarray,
    n_sources: int | None = None,
    *,
    cluster_axis: int = 0,
    aggregation: str = "sum",
    clip: bool = True,
) -> np.ndarray:
    """
    Merge frequency-wise cluster masks into final source masks.

    Parameters
    ----------
    cluster_masks:
        Either shape (C, F, T), matching Sawada masks, or shape (F, C, T) if
        cluster_axis=1.
    labels:
        Source label of each centroid/cluster, shape (F, C). A label of -1 is
        treated as unassigned and contributes to no source.
    aggregation:
        "sum" is appropriate for soft posteriors or disjoint hard masks.
        "max" is an OR-like merge for hard masks.
    clip:
        If True, clip final masks to [0, 1]. This keeps merged hard masks valid
        even when several clusters are assigned to the same source.
    """
    cluster_masks = np.asarray(cluster_masks, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if labels.ndim != 2:
        raise ValueError("labels must have shape (F, C).")
    if cluster_axis == 0:
        if cluster_masks.ndim != 3 or cluster_masks.shape[:2] != (labels.shape[1], labels.shape[0]):
            raise ValueError("cluster_masks must have shape (C, F, T) when cluster_axis=0.")
        masks_cft = cluster_masks
    elif cluster_axis == 1:
        if cluster_masks.ndim != 3 or cluster_masks.shape[:2] != labels.shape:
            raise ValueError("cluster_masks must have shape (F, C, T) when cluster_axis=1.")
        masks_cft = np.moveaxis(cluster_masks, 1, 0)
    else:
        raise ValueError("cluster_axis must be 0 or 1.")

    if aggregation not in {"sum", "max"}:
        raise ValueError("aggregation must be 'sum' or 'max'.")
    if n_sources is None:
        n_sources = int(np.max(labels)) + 1 if np.any(labels >= 0) else 0
    if n_sources < 1:
        return np.empty((0, labels.shape[0], masks_cft.shape[2]), dtype=float)

    source_masks = np.zeros((n_sources, labels.shape[0], masks_cft.shape[2]), dtype=float)
    for source_index in range(n_sources):
        for frequency_index in range(labels.shape[0]):
            selected_clusters = np.flatnonzero(labels[frequency_index] == source_index)
            if selected_clusters.size == 0:
                continue
            selected_masks = masks_cft[selected_clusters, frequency_index, :]
            if aggregation == "sum":
                source_masks[source_index, frequency_index, :] = np.sum(selected_masks, axis=0)
            else:
                source_masks[source_index, frequency_index, :] = np.max(selected_masks, axis=0)

    if clip:
        source_masks = np.clip(source_masks, 0.0, 1.0)
    return source_masks
