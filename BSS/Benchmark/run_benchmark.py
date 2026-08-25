from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .config import BenchmarkConfig
from .io import (
    iter_scenes,
    save_sawada_model_npz,
    save_sources_npz,
    scene_result_dir,
    write_json,
    write_summary_csv,
)
from .runners import run_algorithm
from .tdoa_metrics import (
    align_sources_by_tdoa,
    compute_tdoa_error_metrics,
    pairwise_tdoa_labels,
    true_pairwise_tdoas_samples,
    true_pairwise_tdoas_seconds,
    true_reference_tdoas_samples,
    true_reference_tdoas_seconds,
)


def _pairwise_tdoas_from_ransac_slopes(result: Any) -> np.ndarray:
    sawada_model = result.debug_artifacts.get("sawada_model", {})
    slopes = np.asarray(sawada_model.get("source_assignment_slopes", []), dtype=float)
    if slopes.ndim != 2 or slopes.size == 0:
        return np.empty((0, 0), dtype=float)
    if not np.all(np.isfinite(slopes)):
        return np.empty((0, 0), dtype=float)

    # RANSAC fits arg(C_m * conj(C_1)) = intercept + slope_m * f.
    # For pure delays, slope_m = -2*pi*(delay_m - delay_1).
    relative_delays = -slopes / (2.0 * np.pi)
    n_sources, n_mics = relative_delays.shape
    pairwise = np.empty((n_sources, n_mics * (n_mics - 1) // 2), dtype=float)
    pair_index = 0
    for first in range(n_mics - 1):
        for second in range(first + 1, n_mics):
            pairwise[:, pair_index] = relative_delays[:, second] - relative_delays[:, first]
            pair_index += 1
    return pairwise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Sawada/ICA sur un dataset BSS.")
    parser.add_argument("--dataset", required=True, type=Path, help="Racine du dataset genere.")
    parser.add_argument("--split", default="test", help="Split a traiter: train, validation, test ou all.")
    parser.add_argument("--output", required=True, type=Path, help="Dossier de sortie du benchmark.")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["sawada", "ica"],
        choices=["sawada", "ica"],
        help="Algorithmes a executer.",
    )
    parser.add_argument(
        "--reference-microphone",
        default=0,
        type=int,
        help="Microphone de reference pour les TDOA relatifs stockes en debug.",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Nombre maximal de scenes a traiter, utile pour verifier rapidement.",
    )
    parser.add_argument(
        "--sawada-min-frequency",
        default=None,
        type=float,
        help="Frequence minimale en Hz utilisee par l'EM/RANSAC Sawada.",
    )
    parser.add_argument(
        "--sawada-max-frequency",
        default=None,
        type=float,
        help="Frequence maximale en Hz utilisee par l'EM/RANSAC Sawada.",
    )
    return parser.parse_args()


def _success_summary_row(
    config: BenchmarkConfig,
    record: Any,
    algorithm: str,
    metrics: dict[str, float],
    runtime_seconds: float,
    permutation: tuple[int, ...],
    ransac_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    ransac_metrics = {} if ransac_metrics is None else ransac_metrics
    return {
        "status": "ok",
        "split": record.split,
        "scene_id": record.scene_id,
        "algorithm": algorithm,
        "runtime_seconds": runtime_seconds,
        "source_permutation": list(permutation),
        "sawada_min_frequency_hz": config.sawada_min_frequency_hz,
        "sawada_max_frequency_hz": config.sawada_max_frequency_hz,
        **metrics,
        **{f"ransac_{key}": value for key, value in ransac_metrics.items()},
    }


def _failure_summary_row(
    record: Any,
    algorithm: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "split": record.split,
        "scene_id": record.scene_id,
        "algorithm": algorithm,
        "error": f"{type(error).__name__}: {error}",
    }


def _run_one_algorithm(
    config: BenchmarkConfig,
    record: Any,
    scene: Any,
    algorithm: str,
) -> dict[str, Any]:
    result = run_algorithm(
        algorithm=algorithm,
        record=record,
        scene=scene,
        reference_microphone=config.reference_microphone,
        sawada_min_frequency_hz=config.sawada_min_frequency_hz,
        sawada_max_frequency_hz=config.sawada_max_frequency_hz,
    )

    fs = scene.metadata.fs
    target_pairwise_seconds = true_pairwise_tdoas_seconds(scene)
    target_pairwise_samples = true_pairwise_tdoas_samples(scene)
    alignment = align_sources_by_tdoa(
        estimated_tdoas=result.estimated_tdoas_seconds,
        target_tdoas=target_pairwise_seconds,
        metric="rmse",
    )
    aligned_samples = alignment.aligned_estimated * fs
    metrics = compute_tdoa_error_metrics(
        aligned_estimated_seconds=alignment.aligned_estimated,
        target_seconds=target_pairwise_seconds,
        fs=fs,
    )
    ransac_pairwise_seconds = (
        _pairwise_tdoas_from_ransac_slopes(result)
        if algorithm == "sawada"
        else np.empty((0, 0), dtype=float)
    )
    ransac_alignment = None
    ransac_aligned_seconds = np.empty((0, 0), dtype=float)
    ransac_metrics: dict[str, float] = {}
    if ransac_pairwise_seconds.shape == target_pairwise_seconds.shape:
        ransac_alignment = align_sources_by_tdoa(
            estimated_tdoas=ransac_pairwise_seconds,
            target_tdoas=target_pairwise_seconds,
            metric="rmse",
        )
        ransac_aligned_seconds = ransac_alignment.aligned_estimated
        ransac_metrics = compute_tdoa_error_metrics(
            aligned_estimated_seconds=ransac_aligned_seconds,
            target_seconds=target_pairwise_seconds,
            fs=fs,
        )
    ransac_pairwise_samples = ransac_pairwise_seconds * fs
    ransac_aligned_samples = ransac_aligned_seconds * fs
    labels = pairwise_tdoa_labels(scene.metadata.n_mics)
    output_dir = scene_result_dir(config.output, record.split, record.scene_id)

    save_sources_npz(
        output_dir / f"{algorithm}_sources.npz",
        result=result,
        true_tdoas_seconds=target_pairwise_seconds,
        true_tdoas_samples=target_pairwise_samples,
        aligned_tdoas_seconds=alignment.aligned_estimated,
        aligned_tdoas_samples=aligned_samples,
        pairwise_labels=labels,
    )
    if algorithm == "sawada":
        save_sawada_model_npz(output_dir / "sawada_model.npz", result)

    metrics_payload = {
        "status": "ok",
        "scene_id": record.scene_id,
        "split": record.split,
        "algorithm": algorithm,
        "scene_path": record.path,
        "seed": record.seed,
        "fs": fs,
        "n_sources": scene.metadata.n_sources,
        "n_mics": scene.metadata.n_mics,
        "reference_microphone": config.reference_microphone,
        "max_lag_samples": scene.metadata.max_delay,
        "sawada_frequency_band_hz": {
            "min": config.sawada_min_frequency_hz,
            "max": config.sawada_max_frequency_hz,
        },
        "pairwise_labels": labels,
        "source_permutation": alignment.permutation,
        "alignment_metric": alignment.metric,
        "alignment_score_seconds": alignment.score,
        "runtime_seconds": result.runtime_seconds,
        "parameters": result.parameters,
        "shapes": {
            "estimated_tdoas": result.estimated_tdoas_seconds.shape,
            "true_pairwise_tdoas": target_pairwise_seconds.shape,
            "aligned_tdoas": alignment.aligned_estimated.shape,
            "ransac_pairwise_tdoas": ransac_pairwise_seconds.shape,
        },
        "metrics": metrics,
        "ransac_metrics": ransac_metrics,
        "estimated_pairwise_tdoas_seconds": result.estimated_tdoas_seconds,
        "estimated_pairwise_tdoas_samples": result.estimated_tdoas_samples,
        "aligned_pairwise_tdoas_seconds": alignment.aligned_estimated,
        "aligned_pairwise_tdoas_samples": aligned_samples,
        "ransac_pairwise_tdoas_seconds": ransac_pairwise_seconds,
        "ransac_pairwise_tdoas_samples": ransac_pairwise_samples,
        "ransac_aligned_pairwise_tdoas_seconds": ransac_aligned_seconds,
        "ransac_aligned_pairwise_tdoas_samples": ransac_aligned_samples,
        "ransac_source_permutation": (
            ()
            if ransac_alignment is None
            else ransac_alignment.permutation
        ),
        "ransac_alignment_score_seconds": (
            np.nan
            if ransac_alignment is None
            else ransac_alignment.score
        ),
        "true_pairwise_tdoas_seconds": target_pairwise_seconds,
        "true_pairwise_tdoas_samples": target_pairwise_samples,
        "true_reference_tdoas_seconds": true_reference_tdoas_seconds(
            scene,
            config.reference_microphone,
        ),
        "true_reference_tdoas_samples": true_reference_tdoas_samples(
            scene,
            config.reference_microphone,
        ),
    }
    write_json(output_dir / f"{algorithm}_metrics.json", metrics_payload)
    return _success_summary_row(
        config=config,
        record=record,
        algorithm=algorithm,
        metrics=metrics,
        runtime_seconds=result.runtime_seconds,
        permutation=alignment.permutation,
        ransac_metrics=ransac_metrics,
    )


def run_benchmark(config: BenchmarkConfig) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for record, scene in iter_scenes(
        config.dataset,
        split=config.split,
        limit=config.limit,
    ):
        print(f"Scene {record.split}/{record.scene_id}")
        for algorithm in config.algorithms:
            print(f"  - {algorithm}...", flush=True)
            try:
                row = _run_one_algorithm(config, record, scene, algorithm)
            except Exception as exc:
                row = _failure_summary_row(record, algorithm, exc)
                output_dir = scene_result_dir(config.output, record.split, record.scene_id)
                write_json(
                    output_dir / f"{algorithm}_metrics.json",
                    row,
                )
                print(f"    failed: {exc}")
            else:
                print(
                    "    ok "
                    f"rmse={row['rmse_samples']:.3f} samples "
                    f"mae={row['mae_samples']:.3f} samples"
                )
            summary_rows.append(row)

    write_summary_csv(config.output / "summary.csv", summary_rows)
    return summary_rows


def main() -> None:
    args = parse_args()
    config = BenchmarkConfig(
        dataset=args.dataset,
        split=args.split,
        output=args.output,
        algorithms=tuple(args.algorithms),
        reference_microphone=args.reference_microphone,
        limit=args.limit,
        sawada_min_frequency_hz=args.sawada_min_frequency,
        sawada_max_frequency_hz=args.sawada_max_frequency,
    )
    run_benchmark(config)
    print(f"Resume ecrit dans {(config.output / 'summary.csv').resolve()}")


if __name__ == "__main__":
    main()
