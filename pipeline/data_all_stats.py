"""Genere positions, TDOA et vecteurs directeurs en une seule passe pipeline.

Ce script evite de relancer trois fois la pipeline audio. Pour chaque point, il
appelle une seule fois :

    positions_from_audio(..., return_tdoas=True)

puis il ecrit dans un meme dossier horodate :
    - summary.csv
    - point_<n>_positions_detail.csv
    - plots/point_<n>_positions_vs_ground_truth.png
    - tdoas_detail.csv
    - summary_by_pair.csv
    - directions_detail.csv
    - plots/point_<n>_tdoa_distributions.png

Utilisation simple, depuis le dossier pipeline :
    python data_all_stats.py

Limiter a quelques points :
    python data_all_stats.py --points 7 8

Choisir le dossier de sortie :
    python data_all_stats.py --output-dir "test_data2026_all/results/result_test"
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from legacy_stats_scripts.data_direction_vectors import (
    DIRECTION_FIELDS,
    direction_row,
    group_sort_key,
    local_pair_name,
)
from legacy_stats_scripts.data_position_stats import (
    compute_position_stats,
    plot_positions_and_ground_truth,
    position_times_for_stats,
    print_stats,
    write_position_details_csv,
)
from legacy_stats_scripts.data_tdoa_stats import (
    DETAIL_FIELDS,
    SUMMARY_FIELDS,
    compute_tdoa_stats,
    detail_rows,
    write_csv,
)
from data_paths_2026 import (
    ENV_PATH,
    GROUND_TRUTH_PATH,
    MODEL_PATH,
    PARAM_PATH,
    POINT_NUMBERS,
    TEST_DATA2026_ALL_AUDIO_PATHS,
)
from main_module import positions_from_audio
from plot_tdoa_distributions import (
    DEFAULT_BIN_WIDTH_US,
    plot_distributions,
    read_tdoa_rows,
)
from src.utils.sub_classes import Environment, Parameters
POINT_NUMBERS = [8]
TEST_AUDIO_PATHS = TEST_DATA2026_ALL_AUDIO_PATHS
DEFAULT_RESULTS_ROOT = Path("test_data2026_all") / "results"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calcule positions, TDOA et vecteurs directeurs dans un seul run."
        )
    )
    parser.add_argument(
        "--points",
        type=int,
        nargs="+",
        default=POINT_NUMBERS,
        help=(
            "Liste des points a traiter. "
            f"Par defaut : {' '.join(str(point) for point in POINT_NUMBERS)}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Dossier de sortie. Par defaut : "
            "test_data2026_all/results/result_YYYYMMDD_HHMMSS_micro."
        ),
    )
    parser.add_argument("--model", default=MODEL_PATH, help="Chemin du modele.")
    parser.add_argument(
        "--environment",
        default=ENV_PATH,
        help="Chemin du JSON environnement.",
    )
    parser.add_argument(
        "--parameters",
        default=PARAM_PATH,
        help="Chemin du JSON parametres.",
    )
    parser.add_argument(
        "--ground-truth",
        default=GROUND_TRUTH_PATH,
        help="Chemin du CSV ground truth.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Affiche les plots pendant le run.",
    )
    parser.add_argument(
        "--skip-position-plots",
        action="store_true",
        help="N'ecrit pas les plots de positions.",
    )
    parser.add_argument(
        "--skip-tdoa-plots",
        action="store_true",
        help="N'ecrit pas les histogrammes TDOA dans plots/.",
    )
    parser.add_argument(
        "--include-rejected-tdoa-plots",
        action="store_true",
        help="Inclut aussi usable=False dans les histogrammes TDOA.",
    )
    parser.add_argument(
        "--tdoa-bin-width-us",
        type=float,
        default=DEFAULT_BIN_WIDTH_US,
        help=(
            "Largeur commune des bins TDOA en microsecondes "
            f"({DEFAULT_BIN_WIDTH_US:.6f} par defaut, soit un echantillon a 384 kHz)."
        ),
    )
    return parser


def default_output_dir() -> Path:
    run_id = datetime.now().strftime("result_%Y%m%d_%H%M%S_%f")
    return DEFAULT_RESULTS_ROOT / run_id


def tdoa_groups_from_rows(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, ...], dict[str, dict[str, Any]]]:
    groups: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["point_number"]),
            str(row["frame_index"]),
            str(row["timestamp"]),
            str(row["call_type"]),
            str(row["event_status"]),
            str(row["tetra_id"]),
        )
        groups.setdefault(key, {})[local_pair_name(str(row["pair_id"]))] = row
    return groups


def compute_direction_rows(
    tdoa_rows: list[dict[str, Any]],
    environment: Environment,
) -> list[dict[str, Any]]:
    groups = tdoa_groups_from_rows(tdoa_rows)
    return [
        direction_row(key, pair_rows, environment)
        for key, pair_rows in sorted(groups.items(), key=group_sort_key)
    ]


def write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    failures: list[dict[str, Any]],
) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "points": args.points,
        "model_path": args.model,
        "environment_path": args.environment,
        "parameters_path": args.parameters,
        "ground_truth_path": args.ground_truth,
        "failures_count": len(failures),
        "note": (
            "Positions, TDOA et directions ont ete produits depuis le meme "
            "appel pipeline par point."
        ),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as json_file:
        json.dump(manifest, json_file, indent=2, ensure_ascii=False)


def main() -> None:
    args = build_argument_parser().parse_args()
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=False)

    print(f"Resultats de cette execution : {output_dir}")
    print("Une seule passe pipeline sera lancee par point.")

    parameters = Parameters(args.parameters)
    environment = Environment(
        args.environment,
        parameters.location_parameters.use_h4,
    )

    summaries: list[dict[str, int | float]] = []
    failures: list[dict[str, Any]] = []
    all_tdoa_rows: list[dict[str, Any]] = []

    for point_number in args.points:
        print(f"\n===== Point {point_number} =====")
        audio_paths = TEST_AUDIO_PATHS[point_number] #TEST_DATA2026_ALL_AUDIO_PATHS[point_number]

        try:
            (
                positions,
                errors,
                timestamps,
                durations,
                call_types,
                event_times,
                event_durations,
                event_call_types,
                event_status,
                detections_dfs,
                tdoa_measurements,
            ) = positions_from_audio(
                args.model,
                args.environment,
                args.parameters,
                audio_paths,
                return_tdoas=True,
            )
        except Exception as exc:
            print(f"Point {point_number} ignore : erreur pipeline : {exc}")
            failures.append(
                {
                    "point_number": point_number,
                    "stage": "pipeline",
                    "error": str(exc),
                }
            )
            continue

        point_tdoa_rows = detail_rows(point_number, tdoa_measurements)
        all_tdoa_rows.extend(point_tdoa_rows)
        print(f"{len(point_tdoa_rows)} TDOA extraits pour le point {point_number}")

        try:
            position_times = position_times_for_stats(
                timestamps,
                event_times,
                event_status,
                positions,
            )
            stats = compute_position_stats(
                positions,
                position_times,
                environment,
                args.ground_truth,
            )
        except Exception as exc:
            print(f"Stats position point {point_number} ignorees : {exc}")
            failures.append(
                {
                    "point_number": point_number,
                    "stage": "position_stats",
                    "error": str(exc),
                }
            )
            continue

        print_stats(stats)
        if not args.skip_position_plots:
            plot_positions_and_ground_truth(
                stats,
                environment,
                point_number,
                output_path=str(
                    output_dir
                    / "plots"
                    / f"point_{point_number}_positions_vs_ground_truth.png"
                ),
                show=args.show_plots,
            )

        write_position_details_csv(
            stats,
            output_path=str(output_dir / f"point_{point_number}_positions_detail.csv"),
        )
        summaries.append(
            {
                "point_number": point_number,
                "kept_count": stats["kept_count"],
                "total_count": stats["total_count"],
                "mean_x": stats["mean_xy"][0],
                "mean_y": stats["mean_xy"][1],
                "median_x": stats["median_xy"][0],
                "median_y": stats["median_xy"][1],
                "variance_x": stats["variance_xy"][0],
                "variance_y": stats["variance_xy"][1],
                "std_x": stats["std_xy"][0],
                "std_y": stats["std_xy"][1],
                "std_distance_to_median_xy_m": stats["std_distance_to_median_xy_m"],
                "mean_distance_to_ground_truth_m": stats[
                    "mean_distance_to_ground_truth_m"
                ],
                "mean_distance_to_mean_xy_m": stats["mean_distance_to_mean_xy_m"],
                "mean_distance_to_median_xy_m": stats[
                    "mean_distance_to_median_xy_m"
                ],
            }
        )

    if summaries:
        summary_path = output_dir / "summary.csv"
        with summary_path.open(mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"\nResume positions sauvegarde dans {summary_path}")

    if all_tdoa_rows:
        tdoa_detail_path = output_dir / "tdoas_detail.csv"
        tdoa_summary_path = output_dir / "summary_by_pair.csv"
        write_csv(str(tdoa_detail_path), DETAIL_FIELDS, all_tdoa_rows)
        write_csv(
            str(tdoa_summary_path),
            SUMMARY_FIELDS,
            compute_tdoa_stats(all_tdoa_rows),
        )
        print(f"TDOA detailles sauvegardes dans {tdoa_detail_path}")
        print(f"Statistiques TDOA sauvegardees dans {tdoa_summary_path}")

        direction_rows = compute_direction_rows(all_tdoa_rows, environment)
        directions_path = output_dir / "directions_detail.csv"
        write_csv(str(directions_path), DIRECTION_FIELDS, direction_rows)
        valid_direction_count = sum(
            row["direction_status"] == "ok" for row in direction_rows
        )
        print(f"Directions sauvegardees dans {directions_path}")
        print(
            f"Directions valides : {valid_direction_count} / {len(direction_rows)}"
        )

        if not args.skip_tdoa_plots:
            tdoa_plot_rows = read_tdoa_rows(
                tdoa_detail_path,
                include_rejected=args.include_rejected_tdoa_plots,
            )
            tdoa_plot_paths = plot_distributions(
                tdoa_plot_rows,
                output_dir / "plots",
                include_rejected=args.include_rejected_tdoa_plots,
                image_format="png",
                bin_width_us=args.tdoa_bin_width_us,
            )
            for plot_path in tdoa_plot_paths:
                print(f"Graphique TDOA sauvegarde : {plot_path}")

    if failures:
        failures_path = output_dir / "failures.csv"
        with failures_path.open(mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["point_number", "stage", "error"],
            )
            writer.writeheader()
            writer.writerows(failures)
        print(f"Erreurs sauvegardees dans {failures_path}")

    write_manifest(output_dir, args, failures)
    print(f"Manifest sauvegarde dans {output_dir / 'manifest.json'}")
    print(f"\nTermine. Dossier complet : {output_dir}")


if __name__ == "__main__":
    main()
