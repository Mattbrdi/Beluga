from pathlib import Path
import sys

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from main_module import positions_from_audio
from src.utils.sub_classes import Environment, Parameters
from src.utils.rotation_bricks import lla2enu
from data_paths_2026 import (
    ENV_PATH,
    GROUND_TRUTH_PATH,
    MODEL_PATH,
    PARAM_PATH,
    POINT_NUMBERS,
    TEST_DATA2026_ALL_AUDIO_PATHS,
)
import csv
from datetime import datetime
from math import atan2, cos, radians, sin, sqrt
import os
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer


LatLon = tuple[float, float]
Lla = tuple[float, float, float]


class StatRow(TypedDict):
    index: int
    kept: bool
    datetime: str
    x: float
    y: float
    predicted_lat: float
    predicted_lon: float
    ground_truth_lat: float
    ground_truth_lon: float
    distance_m: float
    distance_to_mean_xy_m: float
    distance_to_median_xy_m: float
    sigma_x: float
    sigma_y: float
    sigma_max: float


class PositionStats(TypedDict):
    mean_xy: np.ndarray
    median_xy: np.ndarray
    variance_xy: np.ndarray
    std_xy: np.ndarray
    std_distance_to_median_xy_m: float
    mean_distance_to_median_xy_m: float
    xy_positions: np.ndarray
    ground_truth_xy: np.ndarray
    sigma_deviation_xy: np.ndarray
    max_sigma_deviation: np.ndarray
    within_two_std: np.ndarray
    kept_count: int
    total_count: int
    mean_distance_to_ground_truth_m: float
    mean_distance_to_mean_xy_m: float
    distances_m: list[float]
    distances_to_mean_xy_m: list[float]
    distances_to_median_xy_m: list[float]
    rows: list[StatRow]


STD_FILTER_THRESHOLD: float = 1.0

ground_truth_path: str = GROUND_TRUTH_PATH
model_path: str = MODEL_PATH
param_path: str = PARAM_PATH
env_path: str = ENV_PATH


def _scalar(x: Any) -> float:
    return float(np.asarray(x).reshape(-1)[0])


def position_xy(position: Any) -> np.ndarray:
    position = np.asarray(position, dtype=float).reshape(-1)
    if position.size < 2:
        raise ValueError(f"Position invalide, au moins 2 coordonnees requises: {position}")
    return position[:2]


def enu_to_lla(
    e: float,
    n: float,
    u: float,
    lat_ref: float,
    lon_ref: float,
    alt_ref: float,
) -> Lla:
    transformer = Transformer.from_crs("epsg:4326", "epsg:4978")
    x_ref, y_ref, z_ref = transformer.transform(lat_ref, lon_ref, alt_ref)

    lat_ref_rad = np.radians(lat_ref)
    lon_ref_rad = np.radians(lon_ref)

    rotation = np.array(
        [
            [-np.sin(lon_ref_rad), np.cos(lon_ref_rad), 0],
            [
                -np.sin(lat_ref_rad) * np.cos(lon_ref_rad),
                -np.sin(lat_ref_rad) * np.sin(lon_ref_rad),
                np.cos(lat_ref_rad),
            ],
            [
                np.cos(lat_ref_rad) * np.cos(lon_ref_rad),
                np.cos(lat_ref_rad) * np.sin(lon_ref_rad),
                np.sin(lat_ref_rad),
            ],
        ]
    )

    d = rotation.T @ np.array([e, n, u], dtype=float)
    x = d[0] + x_ref
    y = d[1] + y_ref
    z = d[2] + z_ref

    transformer = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)
    lon, lat, alt = transformer.transform(x, y, z)
    return float(lat), float(lon), float(alt)


def haversine_m(coord1: LatLon, coord2: LatLon) -> float:
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    radius_m = 6_371_000.0

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return radius_m * c


def lat_long(csv_path: str, datetime_str: str) -> LatLon:
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file, skipinitialspace=True)

        if reader.fieldnames is None:
            raise ValueError(f"CSV vide ou invalide : {csv_path}")

        normalized_fieldnames = [field.strip() for field in reader.fieldnames]
        reader.fieldnames = normalized_fieldnames

        required_columns = {"datetime_correct", "lat", "long"}
        missing_columns = required_columns.difference(normalized_fieldnames)
        if missing_columns:
            raise ValueError(
                f"Colonnes manquantes dans {csv_path}: {', '.join(sorted(missing_columns))}"
            )

        for row in reader:
            normalized_row = {key.strip(): value.strip() for key, value in row.items()}
            if normalized_row["datetime_correct"] == datetime_str:
                return float(normalized_row["lat"]), float(normalized_row["long"])

    raise ValueError(
        f"Aucune ligne trouvee pour datetime_correct = '{datetime_str}' dans {csv_path}"
    )


def format_ground_truth_datetime(timestamp: Any) -> str:
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)
    return timestamp.replace(microsecond=0).strftime("%Y/%m/%d %H:%M:%S")


def position_times_for_stats(
    timestamps: list[Any],
    event_times: list[Any],
    event_status: list[str],
    positions: list[Any],
) -> list[Any]:
    if len(timestamps) == len(positions):
        return timestamps

    ok_event_times = [
        event_time
        for event_time, status in zip(event_times, event_status)
        if status == "ok"
    ]
    if len(ok_event_times) == len(positions):
        return ok_event_times

    raise ValueError(
        "Impossible d'aligner les timestamps avec les positions: "
        f"{len(positions)} positions, {len(timestamps)} timestamps, "
        f"{len(ok_event_times)} event_times ok."
    )


def compute_position_stats(
    positions: list[Any],
    position_times: list[Any],
    environment: Environment,
    ground_truth_csv_path: str,
) -> PositionStats:
    xy_positions: np.ndarray = np.array([position_xy(position) for position in positions], dtype=float)
    valid_mask: np.ndarray = np.all(np.isfinite(xy_positions), axis=1)
    xy_positions = xy_positions[valid_mask]
    position_times = [time for time, is_valid in zip(position_times, valid_mask) if is_valid]

    if len(xy_positions) == 0:
        raise ValueError("Aucune position valide pour calculer les statistiques.")

    mean_xy: np.ndarray = np.mean(xy_positions, axis=0)
    median_xy: np.ndarray = np.median(xy_positions, axis=0)
    variance_xy: np.ndarray = np.var(xy_positions, axis=0)
    std_xy: np.ndarray = np.std(xy_positions, axis=0)

    nonzero_std_axes: np.ndarray = std_xy > 0
    sigma_deviation_xy: np.ndarray = np.zeros_like(xy_positions)
    sigma_deviation_xy[:, nonzero_std_axes] = (
        np.abs(xy_positions[:, nonzero_std_axes] - mean_xy[nonzero_std_axes])
        / std_xy[nonzero_std_axes]
    )
    max_sigma_deviation: np.ndarray = np.max(sigma_deviation_xy, axis=1)

    # On conserve maintenant toutes les positions finies pour garder une
    # analyse transparente. La colonne historique "kept" est gardee pour la
    # compatibilite des CSV/presentations, mais elle vaut toujours True.
    within_two_std = np.ones(len(xy_positions), dtype=bool)

    tetrahedrons_coords: list[LatLon] = [tuple(v.origin_lla[:2]) for v in environment.tetrahedras.values()]
    t1_lat, t1_lon = tetrahedrons_coords[0]

    ground_truth_xy: list[np.ndarray] = []
    rows: list[StatRow] = []
    distances_m: list[float] = []
    distances_to_mean_xy_m: list[float] = []
    distances_to_median_xy_m: list[float] = []
    for i, (xy, timestamp, keep) in enumerate(zip(xy_positions, position_times, within_two_std)):
        predicted_lat_lon: LatLon = enu_to_lla(xy[0], xy[1], 0.0, t1_lat, t1_lon, 0.0)[:2]
        gt_datetime: str = format_ground_truth_datetime(timestamp)
        ground_truth_lat_lon: LatLon = lat_long(ground_truth_csv_path, gt_datetime)
        gt_xy: np.ndarray = lla2enu(
            environment.enu_ref,
            [ground_truth_lat_lon[0], ground_truth_lat_lon[1], 0.0],
        )[:2]
        ground_truth_xy.append(gt_xy)

        distance_m: float = haversine_m(predicted_lat_lon, ground_truth_lat_lon)
        distance_to_mean_xy_m: float = float(np.linalg.norm(xy - mean_xy))
        distance_to_median_xy_m: float = float(np.linalg.norm(xy - median_xy))
        rows.append(
            {
                "index": i,
                "kept": bool(keep),
                "datetime": gt_datetime,
                "x": _scalar(xy[0]),
                "y": _scalar(xy[1]),
                "predicted_lat": predicted_lat_lon[0],
                "predicted_lon": predicted_lat_lon[1],
                "ground_truth_lat": ground_truth_lat_lon[0],
                "ground_truth_lon": ground_truth_lat_lon[1],
                "distance_m": distance_m,
                "distance_to_mean_xy_m": distance_to_mean_xy_m,
                "distance_to_median_xy_m": distance_to_median_xy_m,
                "sigma_x": sigma_deviation_xy[i, 0],
                "sigma_y": sigma_deviation_xy[i, 1],
                "sigma_max": max_sigma_deviation[i],
            }
        )

        distances_m.append(distance_m)
        distances_to_mean_xy_m.append(distance_to_mean_xy_m)
        distances_to_median_xy_m.append(distance_to_median_xy_m)

    if not distances_m:
        raise ValueError("Aucune position valide apres alignement avec la ground truth.")

    return {
        "mean_xy": mean_xy,
        "median_xy": median_xy,
        "variance_xy": variance_xy,
        "std_xy": std_xy,
        "std_distance_to_median_xy_m": float(np.std(distances_to_median_xy_m)),
        "mean_distance_to_median_xy_m": float(np.mean(distances_to_median_xy_m)),
        "xy_positions": xy_positions,
        "ground_truth_xy": np.array(ground_truth_xy, dtype=float),
        "sigma_deviation_xy": sigma_deviation_xy,
        "max_sigma_deviation": max_sigma_deviation,
        "within_two_std": within_two_std,
        "kept_count": int(np.sum(within_two_std)),
        "total_count": int(len(xy_positions)),
        "mean_distance_to_ground_truth_m": float(np.mean(distances_m)),
        "mean_distance_to_mean_xy_m": float(np.mean(distances_to_mean_xy_m)),
        "distances_m": distances_m,
        "distances_to_mean_xy_m": distances_to_mean_xy_m,
        "distances_to_median_xy_m": distances_to_median_xy_m,
        "rows": rows,
    }


def print_stats(stats: PositionStats) -> None:
    print("Statistiques des positions 2D (x, y)")
    print(f"Nombre de positions utilisees: {stats['kept_count']} / {stats['total_count']}")
    print(f"Position moyenne x, y: {stats['mean_xy']}")
    print(f"Position mediane x, y: {stats['median_xy']}")
    print(f"Variance x, y: {stats['variance_xy']}")
    print(f"Ecart type x, y: {stats['std_xy']}")
    print(
        "Distance moyenne a la ground truth: "
        f"{stats['mean_distance_to_ground_truth_m']:.3f} m"
    )
    print(
        "Distance moyenne a la position moyenne: "
        f"{stats['mean_distance_to_mean_xy_m']:.3f} m"
    )
    print(
        "Distance moyenne a la position mediane: "
        f"{stats['mean_distance_to_median_xy_m']:.3f} m"
    )
    print(
        "Ecart type des distances a la position mediane: "
        f"{stats['std_distance_to_median_xy_m']:.3f} m"
    )


def plot_positions_and_ground_truth(
    stats: PositionStats,
    environment: Environment,
    point_number: int,
    output_path: str | None = None,
    show: bool = True,
) -> None:
    xy_positions: np.ndarray = stats["xy_positions"]
    ground_truth_xy: np.ndarray = stats["ground_truth_xy"]
    within_two_std: np.ndarray = stats["within_two_std"]

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(
        xy_positions[within_two_std, 0],
        xy_positions[within_two_std, 1],
        color="tab:blue",
        marker="x",
        s=55,
        label="Positions trouvees",
    )
    ax.plot(
        ground_truth_xy[:, 0],
        ground_truth_xy[:, 1],
        color="tab:orange",
        marker="o",
        markersize=4,
        linewidth=1.5,
        label="Ground truth",
    )
    ax.scatter(
        stats["mean_xy"][0],
        stats["mean_xy"][1],
        color="tab:red",
        marker="+",
        s=180,
        linewidths=2,
        label="Position moyenne",
    )
    ax.scatter(
        stats["median_xy"][0],
        stats["median_xy"][1],
        color="tab:purple",
        marker="D",
        s=90,
        linewidths=1.2,
        edgecolor="black",
        label="Position mediane",
    )

    tetra_x: list[float] = [tetrahedra.origin_enu[0] for tetrahedra in environment.tetrahedras.values()]
    tetra_y: list[float] = [tetrahedra.origin_enu[1] for tetrahedra in environment.tetrahedras.values()]
    ax.scatter(tetra_x, tetra_y, color="tab:green", marker="^", s=70, label="Tetraedres")

    for i, (x, y) in enumerate(xy_positions):
        ax.text(x, y, str(i), fontsize=7, color="black")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x ENU (m)")
    ax.set_ylabel("y ENU (m)")
    ax.set_title(
        f"Point {point_number} - Positions trouvees vs ground truth\n"
        f"Dist. moyenne a la GT: {stats['mean_distance_to_ground_truth_m']:.2f} m | "
        f"Dist. moy. a la moyenne: {stats['mean_distance_to_mean_xy_m']:.2f} m | "
        f"Dist. moy. a la mediane: {stats['mean_distance_to_median_xy_m']:.2f} m"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Plot sauvegarde dans {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_kept_sigma_deviation(
    stats: PositionStats,
    point_number: int,
    output_path: str | None = None,
    show: bool = True,
) -> None:
    within_two_std: np.ndarray = stats["within_two_std"]
    kept_indexes: np.ndarray = np.where(within_two_std)[0]
    kept_sigma_xy: np.ndarray = stats["sigma_deviation_xy"][within_two_std]
    kept_sigma_max: np.ndarray = stats["max_sigma_deviation"][within_two_std]

    fig, ax = plt.subplots(figsize=(10, 5))

    x: np.ndarray = np.arange(len(kept_indexes))
    width: float = 0.25
    ax.bar(x - width, kept_sigma_xy[:, 0], width=width, label="|x - moyenne_x| / std_x")
    ax.bar(x, kept_sigma_xy[:, 1], width=width, label="|y - moyenne_y| / std_y")
    ax.bar(x + width, kept_sigma_max, width=width, label="max axes")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in kept_indexes], rotation=45)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Index de position")
    ax.set_ylabel("Ecart a la moyenne (nombre d'ecarts types)")
    ax.set_title(f"Point {point_number} - Ecarts normalises des positions")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Plot sauvegarde dans {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def write_position_details_csv(stats: PositionStats, output_path: str) -> None:
    fieldnames: list[str] = [
        "index",
        "kept",
        "datetime",
        "x",
        "y",
        "predicted_lat",
        "predicted_lon",
        "ground_truth_lat",
        "ground_truth_lon",
        "distance_m",
        "distance_to_mean_xy_m",
        "distance_to_median_xy_m",
        "sigma_x",
        "sigma_y",
        "sigma_max",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats["rows"])
    print(f"Details des positions sauvegardes dans {output_path}")


def run_point(
    point_number: int,
    parameters: Parameters,
    environment: Environment,
    output_dir: str,
    show_plots: bool = False,
) -> PositionStats:
    audio_path: list[str] = TEST_DATA2026_ALL_AUDIO_PATHS[point_number]
    print(f"\n===== Point {point_number} =====")

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
    ) = positions_from_audio(model_path, env_path, param_path, audio_path)

    position_times: list[Any] = position_times_for_stats(timestamps, event_times, event_status, positions)
    stats: PositionStats = compute_position_stats(
        positions,
        position_times,
        environment,
        ground_truth_path,
    )
    print_stats(stats)

    plot_positions_and_ground_truth(
        stats,
        environment,
        point_number,
        output_path=os.path.join(
            output_dir,
            "plots",
            f"point_{point_number}_positions_vs_ground_truth.png",
        ),
        show=show_plots,
    )
    write_position_details_csv(
        stats,
        output_path=os.path.join(output_dir, f"point_{point_number}_positions_detail.csv"),
    )

    return stats


if __name__ == "__main__":
    parameters: Parameters = Parameters(param_path)
    environment: Environment = Environment(env_path, parameters.location_parameters.use_h4)
    run_id: str = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output_dir: str = os.path.join(
        "test_data2026_all",
        "results",
        "position_stats",
        run_id,
    )
    os.makedirs(output_dir, exist_ok=False)
    print(f"Resultats de cette execution: {output_dir}")

    summaries: list[dict[str, int | float]] = []
    failures: list[dict[str, int | str]] = []
    for point_number in POINT_NUMBERS:
        try:
            stats = run_point(
                point_number,
                parameters,
                environment,
                output_dir,
                show_plots=False,
            )
        except Exception as exc:
            print(f"Point {point_number} ignore a cause d'une erreur: {exc}")
            failures.append({"point_number": point_number, "error": str(exc)})
            continue

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
                "mean_distance_to_ground_truth_m": stats["mean_distance_to_ground_truth_m"],
                "mean_distance_to_mean_xy_m": stats["mean_distance_to_mean_xy_m"],
                "mean_distance_to_median_xy_m": stats["mean_distance_to_median_xy_m"],
            }
        )

    if summaries:
        summary_path: str = os.path.join(output_dir, "summary.csv")
        with open(summary_path, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"\nResume sauvegarde dans {summary_path}")

    if failures:
        failures_path: str = os.path.join(output_dir, "failures.csv")
        with open(failures_path, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["point_number", "error"])
            writer.writeheader()
            writer.writerows(failures)
        print(f"Erreurs sauvegardees dans {failures_path}")
