from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..Algo_Separation.Frequency_ica_separation import FrequencyDomainICA
from ..Algo_Separation.Sawada import SawadaBSS
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


def run_sawada(
    record: Any,
    scene: Any,
    reference_microphone: int = 0,
    min_frequency_hz: float | None = None,
    max_frequency_hz: float | None = None,
) -> SeparationResult:
    start = time.perf_counter()
    max_lag_samples = _max_lag_samples(scene)
    sawada_em_parameters = replace(
        BENCHMARK_SAWADA_EM_PARAMETERS,
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=max_frequency_hz,
    )
    if max_lag_samples is not None and scene.metadata.fs:
        slope_bound = 2.0 * np.pi * float(max_lag_samples) / float(scene.metadata.fs)
        if np.isfinite(slope_bound) and slope_bound > 0.0:
            sawada_em_parameters = replace(
                sawada_em_parameters,
                ransac_slope_bound=1.2 * slope_bound,
            )
    model = SawadaBSS(
        n_sources=scene.metadata.n_sources,
        stft_parameters=BENCHMARK_STFT_PARAMETERS,
        em_clustering_parameters=sawada_em_parameters,
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
                "min_frequency_hz": sawada_em_parameters.min_frequency_hz,
                "max_frequency_hz": sawada_em_parameters.max_frequency_hz,
                "energy_threshold_db_above_floor": (
                    sawada_em_parameters.energy_threshold_db_above_floor
                ),
                "energy_floor_percentile": (
                    sawada_em_parameters.energy_floor_percentile
                ),
                "min_active_frames_per_frequency": (
                    sawada_em_parameters.min_active_frames_per_frequency
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
        debug_artifacts=model.to_debug_artifacts().as_benchmark_artifacts(),
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
    sawada_min_frequency_hz: float | None = None,
    sawada_max_frequency_hz: float | None = None,
) -> SeparationResult:
    if algorithm == "sawada":
        return run_sawada(
            record,
            scene,
            reference_microphone,
            min_frequency_hz=sawada_min_frequency_hz,
            max_frequency_hz=sawada_max_frequency_hz,
        )
    if algorithm == "ica":
        return run_ica(record, scene, reference_microphone)
    raise ValueError(f"Algorithme inconnu: {algorithm}")
