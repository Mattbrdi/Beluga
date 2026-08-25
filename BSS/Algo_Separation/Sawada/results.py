from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SawadaDebugArtifacts:
    """Données optionnelles produites par Sawada pour debug/visualisation."""

    masks: np.ndarray
    posteriors: np.ndarray
    active_clusters: np.ndarray
    cluster_masks: np.ndarray
    cluster_posteriors: np.ndarray
    cluster_active: np.ndarray
    active_frequency_mask: np.ndarray
    bin_vectors: np.ndarray
    tf_energy: np.ndarray
    frequency_energy: np.ndarray
    active_tf_mask: np.ndarray
    energy_threshold_db: np.ndarray
    frequencies: np.ndarray
    times: np.ndarray
    centroids: np.ndarray
    variances: np.ndarray
    weights: np.ndarray
    source_assignment_labels: np.ndarray
    source_assignment_distances: np.ndarray
    source_assignment_relative_phases: np.ndarray
    source_assignment_selected_labels: np.ndarray
    source_assignment_slopes: np.ndarray
    source_assignment_intercepts: np.ndarray
    source_assignment_scores: np.ndarray
    source_assignment_n_inliers: np.ndarray
    source_assignment_n_trials: np.ndarray
    source_assignment_converged: np.ndarray
    source_assignment_frequency_inliers: np.ndarray
    source_assignment_selected_centroids: np.ndarray

    def to_payload(self) -> dict[str, np.ndarray]:
        return {
            "masks": self.masks,
            "posteriors": self.posteriors,
            "active_clusters": self.active_clusters,
            "cluster_masks": self.cluster_masks,
            "cluster_posteriors": self.cluster_posteriors,
            "cluster_active": self.cluster_active,
            "active_frequency_mask": self.active_frequency_mask,
            "bin_vectors": self.bin_vectors,
            "tf_energy": self.tf_energy,
            "frequency_energy": self.frequency_energy,
            "active_tf_mask": self.active_tf_mask,
            "energy_threshold_db": self.energy_threshold_db,
            "frequencies": self.frequencies,
            "times": self.times,
            "centroids": self.centroids,
            "variances": self.variances,
            "weights": self.weights,
            "source_assignment_labels": self.source_assignment_labels,
            "source_assignment_distances": self.source_assignment_distances,
            "source_assignment_relative_phases": self.source_assignment_relative_phases,
            "source_assignment_selected_labels": self.source_assignment_selected_labels,
            "source_assignment_slopes": self.source_assignment_slopes,
            "source_assignment_intercepts": self.source_assignment_intercepts,
            "source_assignment_scores": self.source_assignment_scores,
            "source_assignment_n_inliers": self.source_assignment_n_inliers,
            "source_assignment_n_trials": self.source_assignment_n_trials,
            "source_assignment_converged": self.source_assignment_converged,
            "source_assignment_frequency_inliers": self.source_assignment_frequency_inliers,
            "source_assignment_selected_centroids": self.source_assignment_selected_centroids,
        }

    def as_benchmark_artifacts(self) -> dict[str, dict[str, np.ndarray]]:
        return {"sawada_model": self.to_payload()}


@dataclass(frozen=True)
class SawadaResult:
    """Sortie structurée principale de Sawada."""

    masks: np.ndarray
    posteriors: np.ndarray
    active_clusters: np.ndarray
    centroids: np.ndarray
    frequencies: np.ndarray
    times: np.ndarray
    diagnostics: SawadaDebugArtifacts | None = None
