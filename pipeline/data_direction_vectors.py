"""Calcule une direction ENU par seconde et par tetraedre depuis les TDOA."""

import argparse
import csv
import json
from math import atan2, degrees, sqrt
from pathlib import Path
import re
from typing import Any

import numpy as np

from src.location_bricks.high_level_fusion import direction_vector_from_tdoas
from src.utils.sub_classes import Environment


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TDOA_RESULTS_DIR = SCRIPT_DIR / "test_data2026_all" / "results" / "tdoa_stats"
DEFAULT_ENV_PATH = SCRIPT_DIR / "jsons" / "environments" / "env_cacouna_may2026.json"
DEFAULT_PARAM_PATH = SCRIPT_DIR / "jsons" / "parameters" / "default_parameters.json"
PAIR_PATTERN = re.compile(r"(H\d+H\d+)$")

DIRECTION_FIELDS = [
    "point_number",
    "frame_index",
    "timestamp",
    "call_type",
    "event_status",
    "tetra_id",
    "origin_e_m",
    "origin_n_m",
    "origin_u_m",
    "direction_e",
    "direction_n",
    "direction_u",
    "azimuth_deg",
    "elevation_deg",
    "pair_count_total",
    "pair_count_usable",
    "direction_status",
]


def latest_tdoa_csv(results_dir: Path = DEFAULT_TDOA_RESULTS_DIR) -> Path:
    candidates = list(results_dir.glob("run_*/tdoas_detail.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Aucun tdoas_detail.csv trouve dans {results_dir}. "
            "Lancez d'abord data_tdoa_stats.py."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_input_path(value: str | None) -> Path:
    if value is None:
        return latest_tdoa_csv()
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "tdoas_detail.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Fichier TDOA introuvable : {path}")
    return path


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "oui"}


def local_pair_name(pair_id: str) -> str:
    match = PAIR_PATTERN.search(pair_id.strip())
    return match.group(1) if match else pair_id.strip()


def expected_pair_names(pair_count: int) -> list[str]:
    if pair_count == 6:
        hydrophone_count = 4
    elif pair_count == 3:
        hydrophone_count = 3
    else:
        raise ValueError(f"Nombre de paires non pris en charge : {pair_count}")
    return [
        f"H{i + 1}H{j + 1}"
        for i in range(hydrophone_count)
        for j in range(i + 1, hydrophone_count)
    ]


def read_tdoa_groups(csv_path: Path) -> dict[tuple[str, ...], dict[str, dict[str, Any]]]:
    groups: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for raw_row in reader:
            row = {
                str(key).strip(): value.strip() if isinstance(value, str) else value
                for key, value in raw_row.items()
                if key is not None
            }
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


def direction_row(
    group_key: tuple[str, ...],
    pair_rows: dict[str, dict[str, Any]],
    environment: Environment,
) -> dict[str, Any]:
    point_number, frame_index, timestamp, call_type, event_status, tetra_id = group_key
    base_row: dict[str, Any] = {
        "point_number": point_number,
        "frame_index": frame_index,
        "timestamp": timestamp,
        "call_type": call_type,
        "event_status": event_status,
        "tetra_id": tetra_id,
    }
    tetrahedra = environment.tetrahedras.get(tetra_id)
    if tetrahedra is None:
        return {
            **base_row,
            "origin_e_m": "",
            "origin_n_m": "",
            "origin_u_m": "",
            "direction_e": "",
            "direction_n": "",
            "direction_u": "",
            "azimuth_deg": "",
            "elevation_deg": "",
            "pair_count_total": len(pair_rows),
            "pair_count_usable": 0,
            "direction_status": "unknown_tetrahedron",
        }

    pair_names = expected_pair_names(len(tetrahedra.v_matrix))
    tdoa_values = np.zeros(len(pair_names), dtype=float)
    tdoa_mask = np.zeros(len(pair_names), dtype=bool)
    for index, pair_name in enumerate(pair_names):
        row = pair_rows.get(pair_name)
        if row is None:
            continue
        try:
            tdoa_values[index] = float(row["tdoa_s"])
        except (KeyError, TypeError, ValueError):
            continue
        tdoa_mask[index] = parse_bool(row.get("usable", False))

    origin = np.asarray(tetrahedra.origin_enu, dtype=float).reshape(-1)
    direction = direction_vector_from_tdoas(tdoa_values, tetrahedra, tdoa_mask)
    common = {
        **base_row,
        "origin_e_m": float(origin[0]),
        "origin_n_m": float(origin[1]),
        "origin_u_m": float(origin[2]),
        "pair_count_total": len(pair_rows),
        "pair_count_usable": int(np.sum(tdoa_mask)),
    }
    if direction is None:
        return {
            **common,
            "direction_e": "",
            "direction_n": "",
            "direction_u": "",
            "azimuth_deg": "",
            "elevation_deg": "",
            "direction_status": "insufficient_or_degenerate_tdoas",
        }

    vector = direction.reshape(-1)
    east = float(vector[0])
    north = float(vector[1])
    up = float(vector[2]) if len(vector) == 3 else None
    # Azimut geographique : 0 deg = nord, 90 deg = est.
    azimuth_deg = degrees(atan2(east, north)) % 360.0
    elevation_deg = (
        degrees(atan2(up, sqrt(east**2 + north**2)))
        if up is not None
        else ""
    )
    return {
        **common,
        "direction_e": east,
        "direction_n": north,
        "direction_u": "" if up is None else up,
        "azimuth_deg": azimuth_deg,
        "elevation_deg": elevation_deg,
        "direction_status": "ok",
    }


def load_environment(env_path: Path, param_path: Path) -> Environment:
    with param_path.open(encoding="utf-8") as param_file:
        use_h4 = bool(json.load(param_file)["location_parameters"]["use_h4"])
    return Environment(str(env_path), use_h4)


def group_sort_key(
    item: tuple[tuple[str, ...], dict[str, dict[str, Any]]],
) -> tuple[int, int, str, str, str]:
    key = item[0]
    return int(key[0]), int(key[1]), key[2], key[5], key[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calcule un vecteur direction ENU par seconde et par tetraedre."
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="tdoas_detail.csv ou dossier du run (dernier run par defaut).",
    )
    parser.add_argument("--output", help="Chemin du CSV de sortie.")
    parser.add_argument("--environment", default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--parameters", default=str(DEFAULT_PARAM_PATH))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = resolve_input_path(args.input)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.parent / "directions_detail.csv"
    )
    environment = load_environment(
        Path(args.environment).expanduser().resolve(),
        Path(args.parameters).expanduser().resolve(),
    )
    groups = read_tdoa_groups(input_path)
    rows = [
        direction_row(key, pair_rows, environment)
        for key, pair_rows in sorted(groups.items(), key=group_sort_key)
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=DIRECTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    valid_count = sum(row["direction_status"] == "ok" for row in rows)
    print(f"Directions sauvegardees dans {output_path}")
    print(f"Directions valides : {valid_count} / {len(rows)}")


if __name__ == "__main__":
    main()
