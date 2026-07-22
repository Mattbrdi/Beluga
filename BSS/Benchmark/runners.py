from __future__ import annotations

import time
from dataclasses import dataclass, field
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
    min_active_frames_per_frequency=2,
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

    n_freqs = len(model.bin_models)
    variances = np.zeros((n_freqs, model.n_sources), dtype=float)
    weights = np.zeros((n_freqs, model.n_sources), dtype=float)
    for frequency_index, bin_model in model.bin_models.items():
        variances[frequency_index] = bin_model.variances
        weights[frequency_index] = bin_model.weights

    return {
        "sawada_model": {
            "masks": model.get_final_masks().astype(np.uint8),
            "posteriors": model.get_final_posteriors(),
            "tf_energy": tf_energy,
            "frequency_energy": np.mean(tf_energy, axis=1),
            "active_tf_mask": active_tf_mask.astype(np.uint8),
            "energy_threshold_db": np.asarray(
                np.nan if model.energy_threshold_db is None else model.energy_threshold_db
            ),
            "frequencies": np.asarray(model.nspectro_preprocessed.f, dtype=float),
            "times": np.asarray(model.nspectro_preprocessed.t, dtype=float),
            "centroids": model.all_centroids,
            "variances": variances,
            "weights": weights,
            "whitening": np.asarray(model.whitening),
        }
    }


def run_sawada(
    record: Any,
    scene: Any,
    reference_microphone: int = 0,
) -> SeparationResult:
    start = time.perf_counter()
    max_lag_samples = _max_lag_samples(scene)
    model = SawadaBSS(
        n_sources=scene.metadata.n_sources,
        stft_parameters=BENCHMARK_STFT_PARAMETERS,
        em_clustering_parameters=BENCHMARK_SAWADA_EM_PARAMETERS,
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
            },
            "em_clustering_parameters": {
                "n_iter": BENCHMARK_SAWADA_EM_PARAMETERS.n_iter,
                "phi": BENCHMARK_SAWADA_EM_PARAMETERS.phi,
                "eps": BENCHMARK_SAWADA_EM_PARAMETERS.eps,
                "energy_threshold_db_above_floor": (
                    BENCHMARK_SAWADA_EM_PARAMETERS.energy_threshold_db_above_floor
                ),
                "energy_floor_percentile": (
                    BENCHMARK_SAWADA_EM_PARAMETERS.energy_floor_percentile
                ),
                "min_active_frames_per_frequency": (
                    BENCHMARK_SAWADA_EM_PARAMETERS.min_active_frames_per_frequency
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
