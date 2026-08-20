from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

try:
    from .io import read_manifest
except ImportError:  # pragma: no cover - utile si le fichier est lance directement
    from BSS.Benchmark.io import read_manifest


TABLE_COLUMNS = (
    "Source",
    "Paire",
    "Vrai (samples)",
    "Est. corr. (samples)",
    "Erreur corr. (samples)",
)


@dataclass(frozen=True)
class SceneTdoaTable:
    scene_id: str
    rows: list[list[str]]
    error_labels: list[str]
    errors: np.ndarray
    mae_samples: float
    rmse_samples: float
    max_abs_error_samples: float


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _format_number(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "-"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{digits}f}"


def _pair_labels(metrics: dict[str, Any], n_pairs: int) -> list[str]:
    labels = list(metrics.get("pairwise_labels") or [])
    if len(labels) >= n_pairs:
        return [str(label) for label in labels[:n_pairs]]
    return [f"P{index + 1}" for index in range(n_pairs)]


def _finite_stats(errors: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(errors, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan, math.nan, math.nan
    return (
        float(np.mean(np.abs(finite))),
        float(np.sqrt(np.mean(finite**2))),
        float(np.max(np.abs(finite))),
    )


def _load_scene_table(
    results_root: Path,
    split: str,
    scene_id: str,
    algorithm: str,
    estimation: str,
) -> SceneTdoaTable:
    metrics_path = results_root / split / scene_id / f"{algorithm}_metrics.json"
    metrics = _read_json(metrics_path)
    true_samples = np.asarray(
        metrics.get("true_pairwise_tdoas_samples", []),
        dtype=float,
    )
    estimate_key = (
        "aligned_pairwise_tdoas_samples"
        if estimation == "aligned"
        else "estimated_pairwise_tdoas_samples"
    )
    estimated_samples = np.asarray(metrics.get(estimate_key, []), dtype=float)
    if true_samples.ndim != 2 or estimated_samples.shape != true_samples.shape:
        raise ValueError(
            f"TDOA invalides dans {metrics_path}: true={true_samples.shape}, "
            f"estimated={estimated_samples.shape}."
        )

    errors = estimated_samples - true_samples
    labels = _pair_labels(metrics, true_samples.shape[1])
    rows: list[list[str]] = []
    error_labels: list[str] = []
    for source_index in range(true_samples.shape[0]):
        for pair_index, pair_label in enumerate(labels):
            error_labels.append(f"S{source_index + 1} {pair_label}")
            rows.append(
                [
                    f"S{source_index + 1}",
                    pair_label,
                    _format_number(true_samples[source_index, pair_index]),
                    _format_number(estimated_samples[source_index, pair_index]),
                    _format_number(errors[source_index, pair_index]),
                ]
            )

    mae, rmse, max_abs = _finite_stats(errors)
    return SceneTdoaTable(
        scene_id=scene_id,
        rows=rows,
        error_labels=error_labels,
        errors=errors.reshape(-1),
        mae_samples=mae,
        rmse_samples=rmse,
        max_abs_error_samples=max_abs,
    )


def _error_color(error: float, max_abs_error: float) -> tuple[float, float, float, float]:
    if not np.isfinite(error):
        return (1.0, 1.0, 1.0, 1.0)
    if not np.isfinite(max_abs_error) or max_abs_error <= 0:
        return (1.0, 1.0, 1.0, 1.0)

    normalized = 0.5 + 0.5 * float(error) / float(max_abs_error)
    normalized = min(max(normalized, 0.0), 1.0)
    rgba = plt.get_cmap("coolwarm")(normalized)
    mix = 0.25
    return tuple((1.0 - mix) * channel + mix for channel in rgba[:3]) + (1.0,)


def _draw_scene_table(
    ax: plt.Axes,
    table_data: SceneTdoaTable,
    max_abs_error: float,
    algorithm: str,
    estimation: str,
) -> None:
    ax.axis("off")
    subtitle = (
        f"{algorithm} / correlation {estimation} - "
        f"MAE={_format_number(table_data.mae_samples)} samples, "
        f"RMSE={_format_number(table_data.rmse_samples)} samples"
    )
    ax.set_title(f"{table_data.scene_id}\n{subtitle}", fontsize=12, pad=12)

    table = ax.table(
        cellText=table_data.rows,
        colLabels=TABLE_COLUMNS,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)

    n_columns = len(TABLE_COLUMNS)
    error_column = n_columns - 1
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_linewidth(0.45)
        if row_index == 0:
            cell.set_facecolor("#e8ece8")
            cell.set_text_props(weight="bold")
            continue
        if col_index == error_column:
            error = table_data.errors[row_index - 1]
            cell.set_facecolor(_error_color(error, max_abs_error))
        elif row_index % 2 == 0:
            cell.set_facecolor("#f7f7f3")
        else:
            cell.set_facecolor("#ffffff")


def _load_tables(
    dataset_root: Path,
    results_root: Path,
    split: str,
    algorithm: str,
    max_scenes: int,
    estimation: str,
) -> list[SceneTdoaTable]:
    records = read_manifest(dataset_root, split)[:max_scenes]
    if not records:
        raise ValueError(f"Aucune scene trouvee dans {dataset_root / split}")
    return [
        _load_scene_table(
            results_root,
            split,
            record.scene_id,
            algorithm,
            estimation,
        )
        for record in records
    ]


def _global_max_abs_error(tables: list[SceneTdoaTable]) -> float:
    errors = np.concatenate([table.errors for table in tables if table.errors.size])
    finite = errors[np.isfinite(errors)]
    if finite.size == 0:
        return 1.0
    return max(float(np.max(np.abs(finite))), 1e-9)


def save_pdf(
    tables: list[SceneTdoaTable],
    output: Path,
    algorithm: str,
    estimation: str,
    shared_error_scale: bool,
) -> None:
    global_max_error = _global_max_abs_error(tables)
    with PdfPages(output) as pdf:
        for table_data in tables:
            max_abs_error = (
                global_max_error
                if shared_error_scale
                else max(table_data.max_abs_error_samples, 1e-9)
            )
            fig, ax = plt.subplots(figsize=(11.7, 8.3), constrained_layout=True)
            fig.patch.set_facecolor("#f7f7f3")
            _draw_scene_table(ax, table_data, max_abs_error, algorithm, estimation)
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)


def save_image(
    tables: list[SceneTdoaTable],
    output: Path,
    algorithm: str,
    estimation: str,
    shared_error_scale: bool,
    dpi: int,
    row_height: float,
) -> None:
    n_rows = len(tables)
    global_max_error = _global_max_abs_error(tables)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(12.0, max(row_height * n_rows, 3.0)),
        squeeze=False,
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#f7f7f3")
    for ax, table_data in zip(axes[:, 0], tables):
        max_abs_error = (
            global_max_error
            if shared_error_scale
            else max(table_data.max_abs_error_samples, 1e-9)
        )
        _draw_scene_table(ax, table_data, max_abs_error, algorithm, estimation)

    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_error_heatmap(
    tables: list[SceneTdoaTable],
    output: Path,
    algorithm: str,
    estimation: str,
    shared_error_scale: bool,
    dpi: int,
    fig_width: float,
    row_height: float,
) -> None:
    if not tables:
        raise ValueError("Aucun tableau TDOA a tracer.")

    error_labels = tables[0].error_labels
    for table in tables:
        if table.error_labels != error_labels:
            raise ValueError(
                "Les scenes n'ont pas toutes les memes paires TDOA; "
                "impossible de faire une heatmap unique."
            )

    matrix = np.vstack([table.errors for table in tables])
    if shared_error_scale:
        max_abs_error = _global_max_abs_error(tables)
    else:
        row_max = [
            max(table.max_abs_error_samples, 1e-9)
            for table in tables
        ]
        matrix = matrix / np.asarray(row_max)[:, None]
        max_abs_error = 1.0

    n_rows, n_cols = matrix.shape
    fig, ax = plt.subplots(
        figsize=(fig_width, max(row_height * n_rows, 4.0)),
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#f7f7f3")
    ax.set_facecolor("#ffffff")
    im = ax.imshow(
        matrix,
        cmap="coolwarm",
        vmin=-max_abs_error,
        vmax=max_abs_error,
        aspect="auto",
    )

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(error_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([table.scene_id for table in tables], fontsize=8)
    ax.set_xlabel("Source / paire de micros")
    ax.set_ylabel("Scene")
    title_suffix = "normalisee par scene" if not shared_error_scale else "samples"
    ax.set_title(
        f"Erreurs TDOA correlation - {algorithm} / {estimation} ({title_suffix})",
        fontsize=13,
        pad=14,
    )

    display_matrix = np.vstack([table.errors for table in tables])
    if n_rows * n_cols <= 260:
        for row_index in range(n_rows):
            for col_index in range(n_cols):
                value = display_matrix[row_index, col_index]
                if not np.isfinite(value):
                    label = "-"
                else:
                    label = _format_number(value, digits=1)
                ax.text(
                    col_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="#1f1f1f",
                )

    colorbar_label = (
        "Erreur TDOA corr. (samples)"
        if shared_error_scale
        else "Erreur normalisee par scene"
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label=colorbar_label)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genere une carte d'erreurs TDOA pour le split train: erreurs "
            "d'estimation par correlation en samples, sans RANSAC."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Racine du dataset benchmark.",
    )
    parser.add_argument(
        "--results",
        required=True,
        type=Path,
        help="Racine des resultats benchmark.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--algorithm", default="sawada")
    parser.add_argument("--max-scenes", type=int, default=15)
    parser.add_argument(
        "--view",
        choices=("heatmap", "tables"),
        default="heatmap",
        help="'heatmap' met toutes les scenes sur une image; 'tables' garde les tableaux detailles.",
    )
    parser.add_argument(
        "--estimation",
        choices=("aligned", "raw"),
        default="aligned",
        help=(
            "'aligned' compare apres permutation des sources; 'raw' affiche "
            "l'ordre brut des estimations par correlation."
        ),
    )
    parser.add_argument(
        "--per-scene-error-scale",
        action="store_true",
        help="Utilise une echelle d'erreur separee pour chaque scene.",
    )
    parser.add_argument("--fig-width", type=float, default=13.5)
    parser.add_argument("--row-height", type=float, default=3.4)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    if output is None:
        output = (
            args.results
            / f"{args.split}_{args.algorithm}_tdoa_correlation_errors.png"
        )

    tables = _load_tables(
        dataset_root=args.dataset,
        results_root=args.results,
        split=args.split,
        algorithm=args.algorithm,
        max_scenes=args.max_scenes,
        estimation=args.estimation,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    shared_error_scale = not args.per_scene_error_scale
    if args.view == "heatmap":
        save_error_heatmap(
            tables,
            output,
            args.algorithm,
            args.estimation,
            shared_error_scale,
            args.dpi,
            args.fig_width,
            row_height=0.42,
        )
    elif output.suffix.lower() == ".pdf":
        save_pdf(
            tables,
            output,
            args.algorithm,
            args.estimation,
            shared_error_scale,
        )
    else:
        save_image(
            tables,
            output,
            args.algorithm,
            args.estimation,
            shared_error_scale,
            args.dpi,
            args.row_height,
        )
    print(f"Figure sauvegardee: {output}")


if __name__ == "__main__":
    main()
