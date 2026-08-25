from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from BSS.Benchmark.io import DEFAULT_SPLITS, iter_scene_records, load_scene
    from BSS.SourceCount import estimate_num_sources
except ImportError:  # pragma: no cover
    from ..Benchmark.io import DEFAULT_SPLITS, iter_scene_records, load_scene
    from . import estimate_num_sources


@dataclass(frozen=True)
class EvaluationRow:
    split: str
    scene_id: str
    true_n_sources: int
    estimated_n_sources: int | None
    ok: bool
    valid_frequency_count: int
    active_bin_ratio: float
    median_k: float
    mode_k: int | None
    quantile_k: float


def _keep_consecutive_active_runs(active_frames: np.ndarray, min_run_length: int) -> np.ndarray:
    active_frames = np.asarray(active_frames, dtype=bool)
    if min_run_length <= 1 or active_frames.size == 0:
        return active_frames.copy()

    padded = np.r_[False, active_frames, False]
    transitions = np.diff(padded.astype(int))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    filtered = np.zeros_like(active_frames, dtype=bool)
    for start, stop in zip(starts, stops):
        if stop - start >= min_run_length:
            filtered[start:stop] = True
    return filtered


def _filter_consecutive_runs(mask: np.ndarray, min_run_length: int) -> np.ndarray:
    if min_run_length <= 1:
        return np.asarray(mask, dtype=bool).copy()
    filtered = np.zeros_like(mask, dtype=bool)
    for frequency_index in range(mask.shape[0]):
        filtered[frequency_index] = _keep_consecutive_active_runs(
            mask[frequency_index],
            min_run_length,
        )
    return filtered


def _stft_for_source_count(
    multichannel_data: np.ndarray,
    fs: int,
    nperseg: int,
    noverlap: int | None,
    nfft: int | None,
    window: str,
    boundary: str | None,
    padded: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy import signal as sp_signal

    frequencies, times, stft_values = sp_signal.stft(
        np.asarray(multichannel_data, dtype=float),
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary=boundary,
        padded=padded,
        axis=-1,
    )
    X = np.moveaxis(stft_values, 0, -1)
    return frequencies, times, X


def _build_mask(
    X: np.ndarray,
    mask_mode: str,
    energy_floor_percentile: float,
    energy_threshold_db_above_floor: float,
    min_active_run_length: int,
    max_frequency: float | None,
    frequencies: np.ndarray,
    eps: float,
) -> np.ndarray:
    if mask_mode == "all":
        mask = np.ones(X.shape[:2], dtype=bool)
    elif mask_mode == "energy":
        energy = np.sum(np.abs(X) ** 2, axis=2)
        energy_db = 10.0 * np.log10(energy + eps)
        floor_db = float(np.percentile(energy_db, energy_floor_percentile))
        threshold_db = floor_db + float(energy_threshold_db_above_floor)
        mask = energy_db >= threshold_db
    else:
        raise ValueError(f"Mode de masque inconnu: {mask_mode!r}")

    if max_frequency is not None and max_frequency > 0:
        mask = mask.copy()
        mask[frequencies > max_frequency, :] = False

    return _filter_consecutive_runs(mask, min_active_run_length)


def _mode_int(values: np.ndarray) -> int | None:
    valid = values[np.isfinite(values)].astype(int)
    if valid.size == 0:
        return None
    candidates, counts = np.unique(valid, return_counts=True)
    return int(candidates[np.argmax(counts)])


def _summarize_counts(counts: np.ndarray, quantile: float) -> tuple[float, int | None, float]:
    valid = counts[np.isfinite(counts)]
    if valid.size == 0:
        return np.nan, None, np.nan
    return (
        float(np.median(valid)),
        _mode_int(valid),
        float(np.quantile(valid, quantile)),
    )


def evaluate_scene(record: Any, args: argparse.Namespace) -> EvaluationRow:
    scene = load_scene(record.path)
    frequencies, _, X = _stft_for_source_count(
        scene.mixed.data,
        fs=int(scene.metadata.fs),
        nperseg=int(args.nperseg),
        noverlap=args.noverlap,
        nfft=args.nfft,
        window=args.window,
        boundary=None if args.boundary == "none" else args.boundary,
        padded=not args.no_padded,
    )
    mask = _build_mask(
        X,
        args.mask_mode,
        args.energy_floor_percentile,
        args.energy_threshold_db_above_floor,
        args.min_active_run_length,
        args.max_frequency,
        frequencies,
        args.eps,
    )
    result = estimate_num_sources(
        X,
        mask,
        method=args.method,
        min_selected_frames=args.min_selected_frames,
        relative_threshold=args.relative_threshold,
        min_eigengap_ratio=args.min_eigengap_ratio,
        explained_variance_threshold=args.explained_variance_threshold,
        aggregation=args.aggregation,
        aggregation_quantile=args.aggregation_quantile,
        eps=args.eps,
    )

    estimated = result["estimated_n_sources"]
    true_n_sources = int(scene.metadata.n_sources)
    counts = np.asarray(result["n_sources_per_frequency"], dtype=float)
    median_k, mode_k, quantile_k = _summarize_counts(counts, args.aggregation_quantile)
    valid_frequency_count = int(np.sum(np.asarray(result["valid_frequencies"], dtype=bool)))
    return EvaluationRow(
        split=record.split,
        scene_id=record.scene_id,
        true_n_sources=true_n_sources,
        estimated_n_sources=None if estimated is None else int(estimated),
        ok=estimated == true_n_sources,
        valid_frequency_count=valid_frequency_count,
        active_bin_ratio=float(np.mean(mask)),
        median_k=median_k,
        mode_k=mode_k,
        quantile_k=quantile_k,
    )


def _iter_requested_records(dataset: Path, split: str, max_scenes: int | None):
    count = 0
    splits = DEFAULT_SPLITS if split == "all" else (split,)
    for split_name in splits:
        for record in iter_scene_records(dataset, split_name):
            if max_scenes is not None and count >= max_scenes:
                return
            yield record
            count += 1


def _write_csv(path: Path, rows: list[EvaluationRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(EvaluationRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Teste l'estimateur du nombre de sources sur un dataset benchmark genere "
            "et compare a scene.metadata.n_sources."
        )
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split", default="train", help="train, validation, test ou all.")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)

    parser.add_argument(
        "--method",
        choices=("relative_threshold", "eigengap", "explained_variance"),
        default="explained_variance",
    )
    parser.add_argument(
        "--aggregation",
        choices=("median", "mode", "quantile"),
        default="quantile",
    )
    parser.add_argument("--aggregation-quantile", type=float, default=0.8)
    parser.add_argument("--min-selected-frames", type=int, default=20)
    parser.add_argument("--relative-threshold", type=float, default=0.05)
    parser.add_argument("--min-eigengap-ratio", type=float, default=3.0)
    parser.add_argument("--explained-variance-threshold", type=float, default=0.9)

    parser.add_argument("--mask-mode", choices=("energy", "all"), default="energy")
    parser.add_argument("--energy-floor-percentile", type=float, default=20.0)
    parser.add_argument("--energy-threshold-db-above-floor", type=float, default=6.0)
    parser.add_argument("--min-active-run-length", type=int, default=3)
    parser.add_argument("--max-frequency", type=float, default=None)

    parser.add_argument("--nperseg", type=int, default=4096)
    parser.add_argument("--noverlap", type=int, default=3072)
    parser.add_argument("--nfft", type=int, default=None)
    parser.add_argument("--window", default="hann")
    parser.add_argument(
        "--boundary",
        default="zeros",
        choices=("zeros", "even", "odd", "constant", "none"),
    )
    parser.add_argument("--no-padded", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Retourne un code d'erreur si au moins une scene est mal estimee.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        evaluate_scene(record, args)
        for record in _iter_requested_records(args.dataset, args.split, args.max_scenes)
    ]
    if not rows:
        raise SystemExit(f"Aucune scene trouvee pour {args.dataset} split={args.split}.")

    header = (
        "split scene true est ok valid_freq active_bins median_k mode_k q_k"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        estimated_text = "-" if row.estimated_n_sources is None else str(row.estimated_n_sources)
        mode_text = "-" if row.mode_k is None else str(row.mode_k)
        print(
            f"{row.split:10s} {row.scene_id:12s} "
            f"{row.true_n_sources:4d} {estimated_text:>3s} "
            f"{'OK' if row.ok else 'KO':>2s} "
            f"{row.valid_frequency_count:10d} "
            f"{row.active_bin_ratio:10.1%} "
            f"{row.median_k:8.2f} {mode_text:>6s} {row.quantile_k:6.2f}"
        )

    n_ok = sum(row.ok for row in rows)
    print("-" * len(header))
    print(f"Scenes correctes: {n_ok}/{len(rows)} ({n_ok / len(rows):.1%})")

    if args.output_csv is not None:
        _write_csv(args.output_csv, rows)
        print(f"CSV sauvegarde: {args.output_csv}")

    if args.fail_on_mismatch and n_ok != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
