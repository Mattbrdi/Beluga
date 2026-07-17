"""Exporte les TDOA de la pipeline et leurs statistiques par paire de micros.

Lancer ce script depuis le dossier ``pipeline``. Le CSV de detail contient
toutes les mesures, y compris celles masquees par la pipeline. Les moyennes et
ecarts types sont calcules avec toutes les mesures dont ``usable`` est vrai,
sans exclusion statistique supplementaire.
"""

import csv
from collections import defaultdict
from datetime import datetime
import os
from pathlib import Path
from math import sqrt
from statistics import fmean, median, pstdev
import sys
from typing import Any, Iterable

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

DETAIL_FIELDS = [
    "point_number",
    "frame_index",
    "timestamp",
    "call_type",
    "event_status",
    "duration_s",
    "tetra_id",
    "pair_id",
    "tdoa_s",
    "tdoa_us",
    "error_variance_s2",
    "usable",
]

SUMMARY_FIELDS = [
    "point_number",
    "tetra_id",
    "pair_id",
    "count_total",
    "count_usable",
    "count_rejected",
    "mean_tdoa_s",
    "std_tdoa_s",
    "median_tdoa_s",
    "std_to_median_tdoa_s",
    "mean_tdoa_us",
    "std_tdoa_us",
    "median_tdoa_us",
    "std_to_median_tdoa_us",
]


def two_pass_tdoa_stats(values: Iterable[float]) -> dict[str, Any]:
    """Calcule les statistiques sans retirer de valeurs extremes.

    Le nom de fonction et les cles ``filtered_values`` / ``excluded_count`` sont
    conserves pour compatibilite avec les scripts existants. Desormais,
    ``filtered_values`` contient simplement toutes les valeurs d'entree et
    ``excluded_count`` vaut toujours 0.
    """
    initial_values = [float(value) for value in values]
    if not initial_values:
        return {
            "initial_values": [],
            "filtered_values": [],
            "initial_mean": None,
            "initial_std": None,
            "mean": None,
            "std": None,
            "median": None,
            "std_to_median": None,
            "excluded_count": 0,
        }

    initial_mean = fmean(initial_values)
    initial_std = pstdev(initial_values)
    median_value = median(initial_values)
    std_to_median = sqrt(
        fmean((value - median_value) ** 2 for value in initial_values)
    )
    filtered_values = initial_values.copy()

    return {
        "initial_values": initial_values,
        "filtered_values": filtered_values,
        "initial_mean": initial_mean,
        "initial_std": initial_std,
        "mean": fmean(filtered_values),
        "std": pstdev(filtered_values),
        "median": median_value,
        "std_to_median": std_to_median,
        "excluded_count": 0,
    }


def format_timestamp(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat(sep=" ")
    return str(value)


def detail_rows(
    point_number: int,
    measurements: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for measurement in measurements:
        tdoa_s = float(measurement["tdoa_s"])
        rows.append(
            {
                "point_number": point_number,
                "frame_index": measurement["frame_index"],
                "timestamp": format_timestamp(measurement["timestamp"]),
                "call_type": measurement["call_type"],
                "event_status": measurement["event_status"],
                "duration_s": measurement["duration_s"],
                "tetra_id": measurement["tetra_id"],
                "pair_id": measurement["pair_id"],
                "tdoa_s": tdoa_s,
                "tdoa_us": tdoa_s * 1_000_000.0,
                "error_variance_s2": measurement["error_variance_s2"],
                "usable": bool(measurement["usable"]),
            }
        )
    return rows


def compute_tdoa_stats(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (int(row["point_number"]), str(row["tetra_id"]), str(row["pair_id"]))
        groups[key].append(row)

    summaries = []
    for (point_number, tetra_id, pair_id), group in sorted(groups.items()):
        usable_values = [float(row["tdoa_s"]) for row in group if row["usable"]]
        stats = two_pass_tdoa_stats(usable_values)
        count_usable = len(usable_values)
        mean_s = stats["mean"]
        std_s = stats["std"]
        median_s = stats["median"]
        std_to_median_s = stats["std_to_median"]
        summaries.append(
            {
                "point_number": point_number,
                "tetra_id": tetra_id,
                "pair_id": pair_id,
                "count_total": len(group),
                "count_usable": count_usable,
                "count_rejected": len(group) - count_usable,
                "mean_tdoa_s": mean_s,
                "std_tdoa_s": std_s,
                "median_tdoa_s": median_s,
                "std_to_median_tdoa_s": std_to_median_s,
                "mean_tdoa_us": None if mean_s is None else mean_s * 1_000_000.0,
                "std_tdoa_us": None if std_s is None else std_s * 1_000_000.0,
                "median_tdoa_us": None if median_s is None else median_s * 1_000_000.0,
                "std_to_median_tdoa_us": (
                    None if std_to_median_s is None else std_to_median_s * 1_000_000.0
                ),
            }
        )
    return summaries


def write_csv(path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_point(
    point_number: int,
    audio_paths: list[str],
    model_path: str,
    env_path: str,
    param_path: str,
) -> list[dict[str, Any]]:
    # Import differe pour permettre de reutiliser/tester les fonctions de
    # statistiques sans charger PyTorch et les dependances audio.
    from main_module import tdoas_from_audio

    print(f"\n===== TDOA point {point_number} =====")
    measurements = tdoas_from_audio(
        model_path,
        env_path,
        param_path,
        audio_paths,
    )
    rows = detail_rows(point_number, measurements)
    print(f"{len(rows)} TDOA extraits pour le point {point_number}")
    return rows


if __name__ == "__main__":
    from data_paths_2026 import (
        ENV_PATH,
        MODEL_PATH,
        PARAM_PATH,
        POINT_NUMBERS,
        TEST_DATA2026_ALL_AUDIO_PATHS,
    )

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output_dir = os.path.join(
        "test_data2026_all",
        "results",
        "tdoa_stats",
        run_id,
    )
    os.makedirs(output_dir, exist_ok=False)

    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for current_point in POINT_NUMBERS:
        try:
            all_rows.extend(
                run_point(
                    current_point,
                    TEST_DATA2026_ALL_AUDIO_PATHS[current_point],
                    MODEL_PATH,
                    ENV_PATH,
                    PARAM_PATH,
                )
            )
        except Exception as exc:
            print(f"Point {current_point} ignore a cause d'une erreur: {exc}")
            failures.append({"point_number": current_point, "error": str(exc)})

    detail_path = os.path.join(output_dir, "tdoas_detail.csv")
    summary_path = os.path.join(output_dir, "summary_by_pair.csv")
    write_csv(detail_path, DETAIL_FIELDS, all_rows)
    write_csv(summary_path, SUMMARY_FIELDS, compute_tdoa_stats(all_rows))
    print(f"\nTDOA detailles sauvegardes dans {detail_path}")
    print(f"Statistiques sauvegardees dans {summary_path}")

    if failures:
        failures_path = os.path.join(output_dir, "failures.csv")
        write_csv(failures_path, ["point_number", "error"], failures)
        print(f"Erreurs sauvegardees dans {failures_path}")
