from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..Algo_Separation.Frequency_ica_separation import FrequencyDomainICA
from ..Algo_Separation.Sawada_separation import SawadaBSS
from ..Utils.associated_dataclasses import EMClusteringParameters, StftParameters
from ..Utils.signal_class import MultiSignal


BENCHMARK_STFT_PARAMETERS = StftParameters(
    nperseg=4096,
    noverlap=3072,
)
BENCHMARK_SAWADA_EM_PARAMETERS = EMClusteringParameters(
    n_iter=20,
    phi=1.0,
    eps=1e-12,
    energy_threshold_db_above_floor=6.0,
    energy_floor_percentile=20.0,
    min_active_frames_per_frequency=3,
    merge_centroid_distance_scale=1.0,
    source_alignment_method="ransac",
    ransac_residual_threshold=0.4,
    ransac_max_trials=300,
    ransac_slope_bound=None,
    ransac_random_state=0,
    ransac_local_optimization_steps=1,
    ransac_slope_grid_size=60,
    ransac_n_local_refinements=2,
    ransac_max_hypotheses_per_pair=100,
)


@dataclass(frozen=True)
class SeparationResult:
    algorithm: str
    scene_id: str
    split: str
    sources: list[MultiSignal]
    estimated_tdoas_seconds: np.ndarray
    estimated_tdoas_samples: np.ndarray
    runtime_seconds: float
    parameters: dict[str, Any] = field(default_factory=dict)
    debug_artifacts: dict[str, Any] = field(default_factory=dict)


def _max_lag_samples(scene: Any) -> int | None:
    return None if scene.metadata.max_delay is None else int(scene.metadata.max_delay)


def _result(
    algorithm: str,
    record: Any,
    scene: Any,
    sources: list[MultiSignal],
    estimated_tdoas_seconds: np.ndarray,
    runtime_seconds: float,
    parameters: dict[str, Any],
    debug_artifacts: dict[str, Any] | None = None,
) -> SeparationResult:
    return SeparationResult(
        algorithm=algorithm,
        scene_id=record.scene_id,
        split=record.split,
        sources=sources,
        estimated_tdoas_seconds=np.asarray(estimated_tdoas_seconds, dtype=float),
        estimated_tdoas_samples=np.asarray(estimated_tdoas_seconds, dtype=float)
        * scene.metadata.fs,
        runtime_seconds=runtime_seconds,
        parameters=parameters,
        debug_artifacts={} if debug_artifacts is None else debug_artifacts,
    )


def _sawada_debug_artifacts(model: SawadaBSS) -> dict[str, Any]:
    if model.nspectro_preprocessed is None:
        return {}

    if model.tf_energy is not None:
        tf_energy = model.tf_energy
    elif model.signal is not None:
        raw_spectro = model.get_spectro(model.signal)
        tf_energy = np.sum(np.abs(raw_spectro.Sxx) ** 2, axis=0)
    else:
        tf_energy = np.sum(np.abs(model.nspectro_preprocessed.Sxx) ** 2, axis=0)
    active_tf_mask = (
        model.active_tf_mask
        if model.active_tf_mask is not None
        else np.ones_like(tf_energy, dtype=bool)
    )

    n_freqs = model.nspectro_preprocessed.Sxx.shape[1]
    variances = np.full((n_freqs, model.n_sources), np.nan, dtype=float)
    weights = np.full((n_freqs, model.n_sources), np.nan, dtype=float)
    for frequency_index, bin_model in model.bin_models.items():
        variances[frequency_index] = bin_model.variances
        weights[frequency_index] = bin_model.weights
    centroids = model.all_centroids
    centroids_unwhitened = centroids
    if (
        model.whitening
        and model.nspectro_normalized_unwhitened is not None
        and model.eigenvalues_matrix is not None
        and model.eigenvector_matrix is not None
    ):
        eigenvalues = np.asarray(model.eigenvalues_matrix)
        eigenvectors = np.asarray(model.eigenvector_matrix)
        if eigenvalues.ndim == 2 and eigenvectors.ndim == 3:
            centroids_unwhitened = np.zeros_like(centroids)
            for frequency_index in range(centroids.shape[0]):
                inv_sqrt = 1.0 / np.sqrt(
                    np.maximum(
                        eigenvalues[frequency_index],
                        model.em_clustering_parameters.eps,
                    )
                )
                whitening_matrix = (
                    np.diag(inv_sqrt) @ eigenvectors[frequency_index].conj().T
                )
                centroids_unwhitened[frequency_index] = (
                    np.linalg.pinv(whitening_matrix) @ centroids[frequency_index]
                )
        else:
            whitening_matrix = model.nspectro_normalized_unwhitened.compute_whitening_matrix(
                eigenvalues,
                eigenvectors,
            )
            centroids_unwhitened = np.einsum(
                "ij,fjs->fis",
                np.linalg.pinv(whitening_matrix),
                centroids,
            )
        centroids_unwhitened /= (
            np.linalg.norm(centroids_unwhitened, axis=1, keepdims=True) + 1e-12
        )

    return {
        "sawada_model": {
            "masks": model.get_final_masks().astype(np.uint8),
            "posteriors": model.get_final_posteriors(),
            "active_clusters": model.get_final_active_clusters().astype(np.uint8),
            "active_frequency_mask": (
                np.asarray(model.active_frequency_mask, dtype=np.uint8)
                if model.active_frequency_mask is not None
                else np.ones(n_freqs, dtype=np.uint8)
            ),
            "bin_vectors": model.nspectro_preprocessed.Sxx,
            "bin_vectors_unwhitened": (
                np.empty((0, 0, 0))
                if model.nspectro_normalized_unwhitened is None
                else model.nspectro_normalized_unwhitened.Sxx
            ),
            "tf_energy": tf_energy,
            "frequency_energy": np.mean(tf_energy, axis=1),
            "active_tf_mask": active_tf_mask.astype(np.uint8),
            "energy_threshold_db": np.asarray(
                np.nan if model.energy_threshold_db is None else model.energy_threshold_db
            ),
            "frequencies": np.asarray(model.nspectro_preprocessed.f, dtype=float),
            "times": np.asarray(model.nspectro_preprocessed.t, dtype=float),
            "centroids": centroids,
            "centroids_unwhitened": centroids_unwhitened,
            "variances": variances,
            "weights": weights,
            "whitening": np.asarray(model.whitening),
            "source_assignment_labels": np.asarray(
                model.source_assignment_labels
                if model.source_assignment_labels is not None
                else np.empty((0, 0))
            ),
            "source_assignment_distances": np.asarray(
                model.source_assignment_distances
                if model.source_assignment_distances is not None
                else np.empty((0, 0))
            ),
            "source_assignment_relative_phases": np.asarray(
                model.source_assignment_relative_phases
                if model.source_assignment_relative_phases is not None
                else np.empty((0, 0, 0))
            ),
            "source_assignment_selected_labels": np.asarray(
                model.source_assignment_selected_labels
                if model.source_assignment_selected_labels is not None
                else np.empty((0, 0))
            ),
            "source_assignment_slopes": np.asarray(
                model.source_assignment_slopes
                if model.source_assignment_slopes is not None
                else np.empty((0, 0))
            ),
            "source_assignment_intercepts": np.asarray(
                model.source_assignment_intercepts
                if model.source_assignment_intercepts is not None
                else np.empty((0, 0))
            ),
            "source_assignment_scores": np.asarray(
                model.source_assignment_scores
                if model.source_assignment_scores is not None
                else np.empty((0,))
            ),
            "source_assignment_n_inliers": np.asarray(
                model.source_assignment_n_inliers
                if model.source_assignment_n_inliers is not None
                else np.empty((0,))
            ),
            "source_assignment_n_trials": np.asarray(
                model.source_assignment_n_trials
                if model.source_assignment_n_trials is not None
                else np.empty((0,))
            ),
            "source_assignment_converged": np.asarray(
                model.source_assignment_converged
                if model.source_assignment_converged is not None
                else np.empty((0,))
            ),
            "source_assignment_frequency_inliers": np.asarray(
                model.source_assignment_frequency_inliers
                if model.source_assignment_frequency_inliers is not None
                else np.empty((0, 0))
            ),
            "source_assignment_selected_centroids": np.asarray(
                model.source_assignment_selected_centroids
                if model.source_assignment_selected_centroids is not None
                else np.empty((0, 0))
            ),
        }
    }


def run_sawada(
    record: Any,
    scene: Any,
    reference_microphone: int = 0,
) -> SeparationResult:
    start = time.perf_counter()
    max_lag_samples = _max_lag_samples(scene)
    sawada_em_parameters = BENCHMARK_SAWADA_EM_PARAMETERS
    if max_lag_samples is not None and scene.metadata.fs:
        slope_bound = 2.0 * np.pi * float(max_lag_samples) / float(scene.metadata.fs)
        if np.isfinite(slope_bound) and slope_bound > 0.0:
            sawada_em_parameters = replace(
                BENCHMARK_SAWADA_EM_PARAMETERS,
                ransac_slope_bound=1.2 * slope_bound,
            )
    model = SawadaBSS(
        n_sources=scene.metadata.n_sources,
        stft_parameters=BENCHMARK_STFT_PARAMETERS,
        em_clustering_parameters=sawada_em_parameters,
        whitening=False,
    )
    model.process_signal(scene.mixed)
    sources = model.separate_source()
    estimated_tdoas = model.estimate_pairwise_tdoas(max_lag_samples=max_lag_samples)
    runtime = time.perf_counter() - start
    return _result(
        algorithm="sawada",
        record=record,
        scene=scene,
        sources=sources,
        estimated_tdoas_seconds=estimated_tdoas,
        runtime_seconds=runtime,
        parameters={
            "n_sources": scene.metadata.n_sources,
            "stft_parameters": {
                "nperseg": BENCHMARK_STFT_PARAMETERS.nperseg,
                "noverlap": BENCHMARK_STFT_PARAMETERS.noverlap,
                "nfft": BENCHMARK_STFT_PARAMETERS.nfft,
                "window": BENCHMARK_STFT_PARAMETERS.window,
                "boundary": BENCHMARK_STFT_PARAMETERS.boundary,
                "padded": BENCHMARK_STFT_PARAMETERS.padded,
            },
            "em_clustering_parameters": {
                "n_iter": sawada_em_parameters.n_iter,
                "phi": sawada_em_parameters.phi,
                "eps": sawada_em_parameters.eps,
                "energy_threshold_db_above_floor": (
                    sawada_em_parameters.energy_threshold_db_above_floor
                ),
                "energy_floor_percentile": (
                    sawada_em_parameters.energy_floor_percentile
                ),
                "min_active_frames_per_frequency": (
                    sawada_em_parameters.min_active_frames_per_frequency
                ),
                "merge_centroid_distance_scale": (
                    sawada_em_parameters.merge_centroid_distance_scale
                ),
                "source_alignment_method": sawada_em_parameters.source_alignment_method,
                "ransac_residual_threshold": sawada_em_parameters.ransac_residual_threshold,
                "ransac_max_trials": sawada_em_parameters.ransac_max_trials,
                "ransac_slope_bound": sawada_em_parameters.ransac_slope_bound,
                "ransac_random_state": sawada_em_parameters.ransac_random_state,
                "ransac_local_optimization_steps": (
                    sawada_em_parameters.ransac_local_optimization_steps
                ),
                "ransac_slope_grid_size": sawada_em_parameters.ransac_slope_grid_size,
                "ransac_n_local_refinements": (
                    sawada_em_parameters.ransac_n_local_refinements
                ),
                "ransac_max_hypotheses_per_pair": (
                    sawada_em_parameters.ransac_max_hypotheses_per_pair
                ),
            },
            "reference_microphone": reference_microphone,
            "max_lag_samples": max_lag_samples,
        },
        debug_artifacts=_sawada_debug_artifacts(model),
    )


def run_ica(
    record: Any,
    scene: Any,
    reference_microphone: int = 0,
) -> SeparationResult:
    start = time.perf_counter()
    max_lag_samples = _max_lag_samples(scene)
    model = FrequencyDomainICA(
        n_sources=scene.metadata.n_sources,
        stft_parameters=BENCHMARK_STFT_PARAMETERS,
        reference_microphone=reference_microphone,
        max_lag_samples=max_lag_samples,
    )
    ica_result = model.process_signal(scene.mixed)
    estimated_tdoas = ica_result.pairwise_tdoas
    runtime = time.perf_counter() - start
    return _result(
        algorithm="ica",
        record=record,
        scene=scene,
        sources=ica_result.sources,
        estimated_tdoas_seconds=estimated_tdoas,
        runtime_seconds=runtime,
        parameters={
            "n_sources": scene.metadata.n_sources,
            "stft_parameters": {
                "nperseg": BENCHMARK_STFT_PARAMETERS.nperseg,
                "noverlap": BENCHMARK_STFT_PARAMETERS.noverlap,
                "nfft": BENCHMARK_STFT_PARAMETERS.nfft,
                "window": BENCHMARK_STFT_PARAMETERS.window,
                "boundary": BENCHMARK_STFT_PARAMETERS.boundary,
                "padded": BENCHMARK_STFT_PARAMETERS.padded,
            },
            "reference_microphone": reference_microphone,
            "max_lag_samples": max_lag_samples,
            "max_tdoa_seconds": model.max_tdoa_seconds,
        },
    )


def run_algorithm(
    algorithm: str,
    record: Any,
    scene: Any,
    reference_microphone: int = 0,
) -> SeparationResult:
    if algorithm == "sawada":
        return run_sawada(record, scene, reference_microphone)
    if algorithm == "ica":
        return run_ica(record, scene, reference_microphone)
    raise ValueError(f"Algorithme inconnu: {algorithm}")
