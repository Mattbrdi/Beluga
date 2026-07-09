"""Trace la repartition des TDOA pour chaque point, tetraedre et paire.

Par defaut, le script utilise le ``tdoas_detail.csv`` du run TDOA le plus
recent et ne trace que les mesures marquees comme utilisables.
"""

import argparse
import csv
import math
from pathlib import Path
import re
from statistics import median
from typing import Any

from data_tdoa_stats import two_pass_tdoa_stats


PAIR_PATTERN = re.compile(r"(H\d+H\d+)$")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "test_data2026_all" / "results" / "tdoa_stats"
DEFAULT_BIN_WIDTH_US = 1_000_000.0 / 384_000.0


def latest_detail_csv(results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    candidates = list(results_dir.glob("run_*/tdoas_detail.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Aucun tdoas_detail.csv trouve dans {results_dir}. "
            "Lancez d'abord data_tdoa_stats.py."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_input_path(path: str | None) -> Path:
    if path is None:
        return latest_detail_csv()

    input_path = Path(path).expanduser().resolve()
    if input_path.is_dir():
        input_path = input_path / "tdoas_detail.csv"
    if not input_path.is_file():
        raise FileNotFoundError(f"Fichier TDOA introuvable : {input_path}")
    return input_path


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "oui"}


def pair_name(pair_id: str) -> str:
    match = PAIR_PATTERN.search(pair_id.strip())
    return match.group(1) if match else pair_id.strip()


def pair_sort_key(pair: str) -> tuple[int, int, str]:
    numbers = [int(value) for value in re.findall(r"\d+", pair)]
    if len(numbers) >= 2:
        return numbers[0], numbers[1], pair
    return 999, 999, pair


def point_sort_key(point: str) -> tuple[int, int | str]:
    try:
        return 0, int(point)
    except ValueError:
        return 1, point


def read_tdoa_rows(
    csv_path: Path,
    include_rejected: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for raw_row in reader:
            # Accepte aussi les CSV dont les colonnes/valeurs ont ete alignees
            # manuellement avec des espaces.
            row = {
                str(key).strip(): value.strip() if isinstance(value, str) else value
                for key, value in raw_row.items()
                if key is not None
            }
            usable = parse_bool(str(row.get("usable", "false")))
            if not include_rejected and not usable:
                continue

            try:
                tdoa_us = float(row["tdoa_us"])
            except (KeyError, TypeError, ValueError):
                tdoa_us = float(row["tdoa_s"]) * 1_000_000.0

            if not math.isfinite(tdoa_us):
                continue

            rows.append(
                {
                    "point_number": str(row["point_number"]),
                    "tetra_id": str(row["tetra_id"]),
                    "pair": pair_name(str(row["pair_id"])),
                    "tdoa_us": tdoa_us,
                    "usable": usable,
                }
            )
    return rows


def group_values(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], list[float]]:
    groups: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (row["point_number"], row["tetra_id"], row["pair"])
        groups.setdefault(key, []).append(float(row["tdoa_us"]))
    return groups


def plot_distributions(
    rows: list[dict[str, Any]],
    output_dir: Path,
    include_rejected: bool = False,
    image_format: str = "png",
    bin_width_us: float = DEFAULT_BIN_WIDTH_US,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    initial_groups = group_values(rows)
    group_stats = {
        key: two_pass_tdoa_stats(values)
        for key, values in initial_groups.items()
    }
    groups = {
        key: stats["filtered_values"]
        for key, stats in group_stats.items()
    }
    points = sorted({key[0] for key in groups}, key=point_sort_key)
    if not points:
        raise ValueError("Aucune mesure TDOA a tracer.")
    if bin_width_us <= 0:
        raise ValueError("La largeur des bins doit etre strictement positive.")

    # Une seule grille d'histogramme et les memes limites sont utilisees pour
    # toutes les figures afin de permettre une comparaison visuelle directe.
    all_values = [value for values in groups.values() for value in values]
    max_absolute_tdoa = max(abs(value) for value in all_values)
    x_limit = max(
        bin_width_us,
        math.ceil(max_absolute_tdoa / bin_width_us) * bin_width_us,
    )
    bin_edges = np.arange(
        -x_limit,
        x_limit + bin_width_us * 0.5,
        bin_width_us,
    )
    if bin_edges[-1] < x_limit:
        bin_edges = np.append(bin_edges, x_limit)

    max_bin_count = max(
        int(np.max(np.histogram(values, bins=bin_edges)[0]))
        for values in groups.values()
    )
    y_limit = max(1.0, max_bin_count * 1.10)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for point in points:
        tetra_ids = sorted({key[1] for key in groups if key[0] == point})
        pairs = sorted(
            {key[2] for key in groups if key[0] == point},
            key=pair_sort_key,
        )

        median_std_by_tetra = {
            tetra_id: median(
                stats["std"]
                for (current_point, current_tetra, _), stats in group_stats.items()
                if current_point == point and current_tetra == tetra_id
            )
            for tetra_id in tetra_ids
        }
        tetra_std_title = " | ".join(
            f"tetraedre {tetra_id} : {median_std_by_tetra[tetra_id]:.3f} us"
            for tetra_id in tetra_ids
        )

        title_area_height = 2.2
        figure_height = 3.7 * len(tetra_ids) + title_area_height
        fig, axes = plt.subplots(
            len(tetra_ids),
            len(pairs),
            squeeze=False,
            figsize=(4.0 * len(pairs), figure_height),
        )

        for row_index, tetra_id in enumerate(tetra_ids):
            for column_index, pair in enumerate(pairs):
                ax = axes[row_index][column_index]
                values = groups.get((point, tetra_id, pair), [])
                if not values:
                    ax.axis("off")
                    continue

                stats = group_stats[(point, tetra_id, pair)]
                mean_us = stats["mean"]
                std_us = stats["std"]
                ax.hist(
                    values,
                    bins=bin_edges,
                    color="#174A7E",
                    alpha=1.0,
                    edgecolor="#071E33",
                    linewidth=0.7,
                )
                ax.axvline(
                    mean_us,
                    color="#C1121F",
                    linestyle="--",
                    linewidth=2.0,
                    label="Moyenne",
                )
                ax.set_title(f"{tetra_id} - {pair}")
                ax.set_xlabel("TDOA (microsecondes)")
                ax.set_ylabel("Nombre de mesures")
                ax.set_xlim(-x_limit, x_limit)
                ax.set_ylim(0, y_limit)
                ax.grid(axis="y", alpha=0.25)
                ax.text(
                    0.5,
                    -0.28,
                    f"Moyenne apres filtre : {mean_us:.3f} us\n"
                    f"Ecart type apres filtre : {std_us:.3f} us\n"
                    f"n = {len(values)}/{len(stats['initial_values'])} "
                    f"({stats['excluded_count']} exclue(s))",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=9,
                )

        selection = "toutes les mesures" if include_rejected else "mesures utilisables"
        fig.suptitle(
            f"Point {point} - Repartition des TDOA par tetraedre et paire\n"
            f"Mediane des ecarts types - {tetra_std_title}\n"
            f"({selection}, filtre initial +/- 2 ecarts types, "
            f"bins : {bin_width_us:.3f} us)",
            fontsize=15,
            y=0.98,
        )
        fig.subplots_adjust(
            top=1.0 - title_area_height / figure_height,
            bottom=0.16,
            hspace=0.75,
            wspace=0.32,
        )

        output_path = output_dir / f"point_{point}_tdoa_distributions.{image_format}"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trace les distributions des TDOA par point, tetraedre et paire."
    )
    parser.add_argument(
        "input",
        nargs="?",
        help=(
            "Chemin vers tdoas_detail.csv ou son dossier. "
            "Par defaut, utilise le run le plus recent."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help="Dossier des graphiques (par defaut : <dossier_du_csv>/plots).",
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Inclut aussi les TDOA dont usable=False.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Format des figures (png par defaut).",
    )
    parser.add_argument(
        "--bin-width-us",
        type=float,
        default=DEFAULT_BIN_WIDTH_US,
        help=(
            "Largeur commune des bins en microsecondes "
            f"({DEFAULT_BIN_WIDTH_US:.6f} par defaut, soit un echantillon a 384 kHz)."
        ),
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    input_path = resolve_input_path(args.input)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_path.parent / "plots"
    )
    rows = read_tdoa_rows(input_path, include_rejected=args.include_rejected)
    output_paths = plot_distributions(
        rows,
        output_dir,
        include_rejected=args.include_rejected,
        image_format=args.format,
        bin_width_us=args.bin_width_us,
    )

    print(f"CSV utilise : {input_path}")
    for output_path in output_paths:
        print(f"Graphique sauvegarde : {output_path}")


if __name__ == "__main__":
    main()
