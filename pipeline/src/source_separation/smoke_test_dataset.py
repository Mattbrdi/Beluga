from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PIPELINE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from BSS.Benchmark.io import SceneRecord, load_scene, read_manifest
from src.source_separation.source_separation_gate import (
    MultiTetraSourceSeparationGate,
    SawadaGateConfig,
    SourceCountGateConfig,
)
from src.utils.sub_classes import AudioArray, AudioMetadata


@dataclass
class _SmokeTetrahedra:
    id: str
    max_delay_seconds: float
    rotated_hydro_pos: np.ndarray
    is_active: bool = True
    use_h4: bool = True


@dataclass
class _SmokeEnvironment:
    tetrahedras: dict[str, _SmokeTetrahedra]


def _make_smoke_geometry(n_mics: int, max_delay_seconds: float) -> np.ndarray:
    positions = np.zeros((n_mics, 3), dtype=float)
    if n_mics >= 2:
        positions[1, 0] = 1.0
    if n_mics >= 3:
        positions[2, 1] = 1.0
    if n_mics >= 4:
        positions[3, 2] = 1.0
    return positions * max(max_delay_seconds, 1e-6)


def _make_audio_arrays(scene, n_tetrahedra: int) -> tuple[list[AudioArray], _SmokeEnvironment]:
    fs = int(scene.metadata.fs)
    n_mics = int(scene.metadata.n_mics)
    max_delay_seconds = float(scene.metadata.max_delay) / float(fs)
    geometry = _make_smoke_geometry(n_mics, max_delay_seconds)
    tetrahedras: dict[str, _SmokeTetrahedra] = {}
    audio_arrays: list[AudioArray] = []

    for tetra_index in range(n_tetrahedra):
        tetra_id = f"smoke_T{tetra_index + 1}"
        tetra = _SmokeTetrahedra(
            id=tetra_id,
            max_delay_seconds=max_delay_seconds,
            rotated_hydro_pos=geometry,
        )
        tetrahedras[tetra_id] = tetra
        metadata = AudioMetadata(
            tetra_id=tetra_id,
            beluga_call_type="Whistle",
            call_duration=float(scene.metadata.duration),
            start_time=0.0,
            snr_power=None,
            sample_rate=fs,
            frequency_range=(500.0, min(25000.0, fs / 2.0)),
            central_frequency=None,
        )
        audio_arrays.append(
            AudioArray(
                metadata,
                tetra,
                use_h4=True,
                data_array=np.asarray(scene.mixed.data, dtype=float).copy(),
            )
        )

    return audio_arrays, _SmokeEnvironment(tetrahedras)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Teste MultiTetraSourceSeparationGate sur une ou plusieurs scenes "
            "d'un dataset benchmark BSS."
        )
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument(
        "--all-scenes",
        action="store_true",
        help="Teste toutes les scenes du split au lieu de --scene-index.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Limite le nombre de scenes si --all-scenes est active.",
    )
    parser.add_argument("--n-tetrahedra", type=int, default=2)
    parser.add_argument(
        "--run-sawada",
        action="store_true",
        help="Lance aussi Sawada si le gate decide que la separation est utile.",
    )
    parser.add_argument("--method", default="explained_variance")
    parser.add_argument("--aggregation", default="quantile")
    parser.add_argument("--aggregation-quantile", type=float, default=0.8)
    parser.add_argument("--explained-variance-threshold", type=float, default=0.9)
    parser.add_argument("--min-selected-frames", type=int, default=20)
    parser.add_argument("--min-valid-frequencies", type=int, default=2)
    parser.add_argument("--energy-floor-percentile", type=float, default=20.0)
    parser.add_argument("--energy-threshold-db-above-floor", type=float, default=6.0)
    parser.add_argument("--min-active-run-length", type=int, default=3)
    parser.add_argument("--max-frequency", type=float, default=None)
    parser.add_argument("--min-reliable-tetrahedra", type=int, default=None)
    parser.add_argument("--fail-on-wrong-count", action="store_true")
    return parser.parse_args()


def _run_one_scene(args: argparse.Namespace, record: SceneRecord) -> bool:
    scene = load_scene(record.path)
    audio_arrays, environment = _make_audio_arrays(scene, args.n_tetrahedra)
    source_count_config = SourceCountGateConfig(
        method=args.method,
        aggregation=args.aggregation,
        aggregation_quantile=args.aggregation_quantile,
        explained_variance_threshold=args.explained_variance_threshold,
        min_selected_frames=args.min_selected_frames,
        min_valid_frequencies=args.min_valid_frequencies,
        energy_floor_percentile=args.energy_floor_percentile,
        energy_threshold_db_above_floor=args.energy_threshold_db_above_floor,
        min_active_run_length=args.min_active_run_length,
        max_frequency=args.max_frequency,
    )
    sawada_config = SawadaGateConfig(
        enabled=args.run_sawada,
        min_reliable_tetrahedra=(
            args.min_reliable_tetrahedra
            if args.min_reliable_tetrahedra is not None
            else min(2, args.n_tetrahedra)
        ),
    )
    gate = MultiTetraSourceSeparationGate(source_count_config, sawada_config)
    decision = gate.process(audio_arrays, environment)

    true_n_sources = int(scene.metadata.n_sources)
    count_ok = decision.global_n_sources == true_n_sources
    verdict = "OK" if count_ok else "KO"

    print(f"Scene: {args.split}/{record.scene_id}")
    print(f"Vrai nombre de sources: {true_n_sources}")
    print(f"Nombre global estime: {decision.global_n_sources}")
    print(f"Verdict estimation: {verdict}")
    print(f"Decision separation: {decision.should_separate}")
    print(f"Raison: {decision.reason}")
    print("")
    print("Par tetra:")
    for count in decision.source_counts:
        print(
            f"  {count.tetra_id}: k={count.estimated_n_sources}, "
            f"reliable={count.reliable}, valid_freq={count.valid_frequency_count}, "
            f"active_bins={count.active_bin_ratio:.1%}, reason={count.reason}"
        )

    if decision.should_separate:
        print("")
        print("Sources separees:")
        for source_index, source_audio_arrays in enumerate(
            decision.separated_audio_arrays_by_source,
            start=1,
        ):
            shapes = [item.data_array.shape for item in source_audio_arrays]
            print(f"  Source {source_index}: {len(source_audio_arrays)} tetras, shapes={shapes}")
    print("-" * 72)

    return count_ok


def main() -> None:
    args = parse_args()
    records = read_manifest(args.dataset, args.split)
    if not records:
        raise SystemExit(f"Aucune scene dans {args.dataset / args.split}.")

    if args.all_scenes:
        selected_records = records[: args.max_scenes]
    else:
        if args.scene_index < 0 or args.scene_index >= len(records):
            raise SystemExit(
                f"scene-index invalide: {args.scene_index}, "
                f"nombre de scenes={len(records)}."
            )
        selected_records = [records[args.scene_index]]

    ok_count = 0
    for record in selected_records:
        ok_count += int(_run_one_scene(args, record))

    total = len(selected_records)
    print(f"Resume: {ok_count}/{total} estimations correctes.")

    if args.fail_on_wrong_count and ok_count != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
