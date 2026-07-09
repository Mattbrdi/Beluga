"""
Generation de la presentation HTML des resultats de localisation.

Utilisation simple, depuis le dossier pipeline :
    python generate_position_stats_presentation.py

Par defaut, le script utilise automatiquement :
    - le dernier dossier test_data2026_all/results/position_stats/run_* contenant summary.csv
    - le dernier dossier test_data2026_all/results/tdoa_stats/run_* contenant plots/
    - le dernier dossier test_data2026_all/results/tdoa_stats/run_* contenant directions_detail.csv

Le fichier de presentation est genere dans :
    test_data2026_all/results/

avec un nom unique horodate, donc l'ancienne presentation n'est pas ecrasee.
Les images sont integrees directement dans le HTML, ce qui evite les problemes
de chemins relatifs si le fichier est deplace ou envoye seul.

Pour choisir explicitement les resultats a utiliser :
    python generate_position_stats_presentation.py ^
      --position-results "test_data2026_all/results/position_stats/run_YYYYMMDD_HHMMSS_xxxxxx" ^
      --tdoa-results "test_data2026_all/results/tdoa_stats/run_YYYYMMDD_HHMMSS_xxxxxx" ^
      --direction-results "test_data2026_all/results/tdoa_stats/run_YYYYMMDD_HHMMSS_xxxxxx" ^
      --output "presentation_projection_xy.html"

Sous PowerShell, utiliser plutot le backtick ` en fin de ligne, ou tout mettre
sur une seule ligne.

Pour desactiver les slides TDOA ou directions :
    python generate_position_stats_presentation.py --tdoa-results none --direction-results none

Si --output pointe vers un fichier qui existe deja, le script refuse de
l'ecraser. Ajouter --overwrite uniquement si l'ecrasement est voulu.
"""

import argparse
import base64
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.rotation_bricks import lla2enu
from src.utils.sub_classes import Environment


SCRIPT_DIR = Path(__file__).resolve().parent
ALL_RESULTS_ROOT = SCRIPT_DIR / "test_data2026_all" / "results"
RESULTS_ROOT = ALL_RESULTS_ROOT / "position_stats"
TDOA_RESULTS_ROOT = ALL_RESULTS_ROOT / "tdoa_stats"
ENV_PATH = SCRIPT_DIR / "jsons" / "environments" / "env_cacouna_may2026.json"
PRESENTATION_ENU_SIZE_M = 3_000.0
PRESENTATION_TETRA_MARGIN_M = 250.0
PRESENTATION_DIRECTION_LENGTH_M = PRESENTATION_ENU_SIZE_M * 1.6


def find_results_dir() -> Path:
    run_dirs = [
        path
        for path in RESULTS_ROOT.glob("run_*")
        if path.is_dir() and (path / "summary.csv").exists()
    ]
    return max(run_dirs, key=lambda path: path.stat().st_mtime) if run_dirs else RESULTS_ROOT


def find_tdoa_results_dir() -> Path | None:
    run_dirs = [
        path
        for path in TDOA_RESULTS_ROOT.glob("run_*")
        if path.is_dir() and (path / "plots").is_dir()
    ]
    return max(run_dirs, key=lambda path: path.stat().st_mtime) if run_dirs else None


def find_direction_results_dir() -> Path | None:
    run_dirs = [
        path
        for path in TDOA_RESULTS_ROOT.glob("run_*")
        if path.is_dir() and (path / "directions_detail.csv").is_file()
    ]
    return max(run_dirs, key=lambda path: path.stat().st_mtime) if run_dirs else None


RESULTS_DIR = find_results_dir()
TDOA_RESULTS_DIR = find_tdoa_results_dir()
DIRECTION_RESULTS_DIR = find_direction_results_dir()
SUMMARY_PATH = RESULTS_DIR / "summary.csv"


def default_output_path() -> Path:
    run_name = RESULTS_DIR.name if RESULTS_DIR != RESULTS_ROOT else "root"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ALL_RESULTS_ROOT / f"position_stats_presentation_{run_name}_{timestamp}.html"


OUTPUT_PATH = default_output_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genere la presentation HTML des statistiques de position."
    )
    parser.add_argument(
        "--position-results",
        type=Path,
        default=None,
        help=(
            "Dossier de resultats position_stats a utiliser, contenant summary.csv. "
            "Par defaut: dernier run_* disponible."
        ),
    )
    parser.add_argument(
        "--tdoa-results",
        type=Path,
        default=None,
        help=(
            "Dossier de resultats tdoa_stats a utiliser, contenant plots/. "
            "Par defaut: dernier run_* disponible. Utilise 'none' pour desactiver."
        ),
    )
    parser.add_argument(
        "--direction-results",
        type=Path,
        default=None,
        help=(
            "Dossier de resultats tdoa_stats contenant directions_detail.csv. "
            "Par defaut: dernier run_* disponible. Utilise 'none' pour desactiver."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Chemin du fichier HTML a creer. "
            "Si le chemin est relatif, il est interprete depuis test_data2026_all/results."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Autorise l'ecrasement du fichier --output s'il existe deja.",
    )
    return parser.parse_args()


def resolve_input_dir(path_arg: Path | None) -> Path | None:
    if path_arg is None:
        return None
    if str(path_arg).strip().lower() == "none":
        return None
    return path_arg if path_arg.is_absolute() else (Path.cwd() / path_arg).resolve()


def resolve_output_path(output_arg: Path | None) -> Path:
    if output_arg is None:
        return default_output_path()
    if output_arg.is_absolute():
        return output_arg
    return ALL_RESULTS_ROOT / output_arg


def validate_results_dirs() -> None:
    if not SUMMARY_PATH.is_file():
        raise FileNotFoundError(
            f"Le dossier de positions {RESULTS_DIR} ne contient pas summary.csv."
        )
    if TDOA_RESULTS_DIR is not None and not (TDOA_RESULTS_DIR / "plots").is_dir():
        raise FileNotFoundError(
            f"Le dossier TDOA {TDOA_RESULTS_DIR} ne contient pas de dossier plots/."
        )
    if DIRECTION_RESULTS_DIR is not None and not (
        DIRECTION_RESULTS_DIR / "directions_detail.csv"
    ).is_file():
        raise FileNotFoundError(
            f"Le dossier directions {DIRECTION_RESULTS_DIR} ne contient pas directions_detail.csv."
        )


def fmt(value: float, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}"


def pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def read_detail(point_number: int) -> pd.DataFrame:
    detail_path = RESULTS_DIR / f"point_{point_number}_positions_detail.csv"
    detail_df = pd.read_csv(detail_path, skipinitialspace=True)
    detail_df.columns = [column.strip() for column in detail_df.columns]
    if "kept" in detail_df.columns:
        detail_df["kept"] = detail_df["kept"].astype(str).str.strip().str.lower().eq("true")
    return detail_df


def generate_presentation_enu_plot(
    point_number: int,
    environment: Environment,
) -> Path:
    """Cree le plan ENU recadre utilise uniquement dans la presentation."""
    detail_df = read_detail(point_number)
    xy_positions = detail_df[["x", "y"]].to_numpy(dtype=float)
    kept_mask = detail_df["kept"].to_numpy(dtype=bool)
    ground_truth_xy = np.array(
        [
            lla2enu(
                environment.enu_ref,
                [float(row["ground_truth_lat"]), float(row["ground_truth_lon"]), 0.0],
            )[:2]
            for _, row in detail_df.iterrows()
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(
        xy_positions[:, 0],
        xy_positions[:, 1],
        color="tab:gray",
        marker="x",
        s=35,
        label="Positions trouvees",
    )
    ax.scatter(
        xy_positions[kept_mask, 0],
        xy_positions[kept_mask, 1],
        color="tab:blue",
        marker="x",
        s=55,
        label="Positions gardees",
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
    mean_xy = np.mean(xy_positions[kept_mask], axis=0) if np.any(kept_mask) else np.mean(xy_positions, axis=0)
    ax.scatter(
        mean_xy[0],
        mean_xy[1],
        color="tab:red",
        marker="+",
        s=180,
        linewidths=2,
        label="Position moyenne gardee",
    )

    tetra_ids = list(environment.tetrahedras.keys())
    tetra_x = [float(tetra.origin_enu[0]) for tetra in environment.tetrahedras.values()]
    tetra_y = [float(tetra.origin_enu[1]) for tetra in environment.tetrahedras.values()]
    ax.scatter(
        tetra_x,
        tetra_y,
        color="tab:green",
        marker="^",
        s=80,
        label="Tetraedres",
    )
    for tetra_id, x, y in zip(tetra_ids, tetra_x, tetra_y):
        ax.annotate(
            tetra_id,
            (x, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            color="tab:green",
            fontweight="bold",
        )

    # Le cadrage est propre a la presentation : les donnees et les graphiques
    # originaux produits par data_position_stats.py restent inchanges.
    x_max = max(tetra_x) + PRESENTATION_TETRA_MARGIN_M
    x_min = x_max - PRESENTATION_ENU_SIZE_M
    y_min = min(tetra_y) - PRESENTATION_TETRA_MARGIN_M
    y_max = y_min + PRESENTATION_ENU_SIZE_M
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x ENU (m)")
    ax.set_ylabel("y ENU (m)")
    ax.set_title(f"Point {point_number} - Plan ENU 2 500 m x 2 500 m")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path = RESULTS_DIR / f"point_{point_number}_presentation_enu.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def read_directions() -> pd.DataFrame:
    if DIRECTION_RESULTS_DIR is None:
        return pd.DataFrame()
    directions_df = pd.read_csv(
        DIRECTION_RESULTS_DIR / "directions_detail.csv",
        skipinitialspace=True,
    )
    directions_df.columns = [column.strip() for column in directions_df.columns]
    for column in directions_df.select_dtypes(include="object").columns:
        directions_df[column] = directions_df[column].str.strip()
    directions_df["timestamp_second"] = directions_df["timestamp"].map(
        lambda value: datetime.fromisoformat(str(value).strip()).strftime(
            "%Y/%m/%d %H:%M:%S"
        )
    )
    return directions_df


def generate_direction_map(
    point_number: int,
    environment: Environment,
    directions_df: pd.DataFrame,
) -> Path | None:
    """Trace les directions liees aux positions conservees du point."""
    detail_df = read_detail(point_number)
    kept_df = detail_df[detail_df["kept"]].copy()
    if kept_df.empty or directions_df.empty:
        return None

    kept_times = set(kept_df["datetime"].astype(str).str.strip())
    point_directions = directions_df[
        (directions_df["point_number"].astype(int) == point_number)
        & (directions_df["event_status"] == "ok")
        & (directions_df["direction_status"] == "ok")
        & (directions_df["timestamp_second"].isin(kept_times))
    ].copy()
    if point_directions.empty:
        return None

    kept_xy = kept_df[["x", "y"]].to_numpy(dtype=float)
    kept_ground_truth_xy = np.array(
        [
            lla2enu(
                environment.enu_ref,
                [float(row["ground_truth_lat"]), float(row["ground_truth_lon"]), 0.0],
            )[:2]
            for _, row in kept_df.iterrows()
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(
        kept_xy[:, 0],
        kept_xy[:, 1],
        color="tab:blue",
        marker="x",
        s=55,
        label=f"Positions gardees ({len(kept_df)})",
        zorder=4,
    )
    ax.plot(
        kept_ground_truth_xy[:, 0],
        kept_ground_truth_xy[:, 1],
        color="tab:orange",
        marker="o",
        markersize=4,
        linewidth=1.5,
        label="Ground truth des positions gardees",
        zorder=3,
    )

    tetra_colors = ["#1976D2", "#D32F2F", "#7B1FA2", "#388E3C"]
    tetra_x = []
    tetra_y = []
    for tetra_index, (tetra_id, tetrahedra) in enumerate(environment.tetrahedras.items()):
        origin_e = float(tetrahedra.origin_enu[0])
        origin_n = float(tetrahedra.origin_enu[1])
        tetra_x.append(origin_e)
        tetra_y.append(origin_n)
        tetra_directions = point_directions[
            point_directions["tetra_id"].astype(str) == str(tetra_id)
        ]
        color = tetra_colors[tetra_index % len(tetra_colors)]
        if not tetra_directions.empty:
            ax.quiver(
                np.full(len(tetra_directions), origin_e),
                np.full(len(tetra_directions), origin_n),
                tetra_directions["direction_e"].to_numpy(dtype=float)
                * PRESENTATION_DIRECTION_LENGTH_M,
                tetra_directions["direction_n"].to_numpy(dtype=float)
                * PRESENTATION_DIRECTION_LENGTH_M,
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.0025,
                alpha=0.28,
                color=color,
                label=f"Directions {tetra_id} ({len(tetra_directions)})",
                zorder=2,
            )
        ax.scatter(
            [origin_e],
            [origin_n],
            color=color,
            marker="^",
            s=100,
            edgecolor="black",
            linewidth=0.6,
            zorder=5,
        )
        ax.annotate(
            str(tetra_id),
            (origin_e, origin_n),
            xytext=(7, 7),
            textcoords="offset points",
            color=color,
            fontweight="bold",
        )

    x_max = max(tetra_x) + PRESENTATION_TETRA_MARGIN_M
    x_min = x_max - PRESENTATION_ENU_SIZE_M
    y_min = min(tetra_y) - PRESENTATION_TETRA_MARGIN_M
    y_max = y_min + PRESENTATION_ENU_SIZE_M
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x ENU (m)")
    ax.set_ylabel("y ENU (m)")
    ax.set_title(
        f"Point {point_number} - Directions liees aux positions gardees\n"
        f"Projection horizontale, fleches de {PRESENTATION_DIRECTION_LENGTH_M:.0f} m"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()

    output_path = RESULTS_DIR / f"point_{point_number}_direction_map.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def add_kept_stats(summary_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = summary_df.copy()
    std_x_kept = []
    std_y_kept = []

    for _, row in summary_df.iterrows():
        point_number = int(row["point_number"])
        detail_df = read_detail(point_number)
        kept_df = detail_df[detail_df["kept"]]
        std_x_kept.append(float(kept_df["x"].std(ddof=0)))
        std_y_kept.append(float(kept_df["y"].std(ddof=0)))

    summary_df["std_x_kept"] = std_x_kept
    summary_df["std_y_kept"] = std_y_kept
    return summary_df


def metric_card(label: str, value: str) -> str:
    return f"""
    <div class="metric">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
    </div>
    """


def summary_table(summary_df: pd.DataFrame) -> str:
    rows = []
    for _, row in summary_df.iterrows():
        kept_ratio = row["kept_count"] / row["total_count"]
        rows.append(
            "<tr>"
            f"<td>{int(row['point_number'])}</td>"
            f"<td>{int(row['kept_count'])}/{int(row['total_count'])}</td>"
            f"<td>{pct(kept_ratio)}</td>"
            f"<td>{fmt(row['mean_distance_to_ground_truth_m'])} m</td>"
            f"<td>{fmt(row['mean_distance_to_mean_xy_m'])} m</td>"
            f"<td>({fmt(row['std_x_kept'])}, {fmt(row['std_y_kept'])}) m</td>"
            "</tr>"
        )

    return """
    <table>
      <thead>
        <tr>
          <th>Point</th>
          <th>Gardees</th>
          <th>Taux</th>
          <th>Dist. moy. GT</th>
          <th>Dist. moy. moyenne</th>
          <th>Std x,y gardes</th>
        </tr>
      </thead>
      <tbody>
    """ + "\n".join(rows) + """
      </tbody>
    </table>
    """


def point_slide(row: pd.Series) -> str:
    point_number = int(row["point_number"])
    detail_df = read_detail(point_number)
    kept_df = detail_df[detail_df["kept"]]
    kept_ratio = row["kept_count"] / row["total_count"]

    median_gt_distance = kept_df["distance_m"].median()
    max_gt_distance = kept_df["distance_m"].max()
    median_mean_distance = kept_df["distance_to_mean_xy_m"].median()
    max_sigma = kept_df["sigma_max"].max()

    position_img = image_src(
        RESULTS_DIR / f"point_{point_number}_presentation_enu.png"
    )
    metrics = "\n".join(
        [
            metric_card("Positions gardees", f"{int(row['kept_count'])}/{int(row['total_count'])} ({pct(kept_ratio)})"),
            metric_card("Distance moyenne a la GT", f"{fmt(row['mean_distance_to_ground_truth_m'])} m"),
            metric_card("Distance mediane a la GT", f"{fmt(median_gt_distance)} m"),
            metric_card("Distance max a la GT", f"{fmt(max_gt_distance)} m"),
            metric_card("Distance moyenne a la moyenne", f"{fmt(row['mean_distance_to_mean_xy_m'])} m"),
            metric_card("Distance mediane a la moyenne", f"{fmt(median_mean_distance)} m"),
            metric_card("Std x,y gardes", f"{fmt(row['std_x_kept'])} m, {fmt(row['std_y_kept'])} m"),
            metric_card("Sigma max garde", fmt(max_sigma, 2)),
        ]
    )

    return f"""
    <section class="slide point-slide">
      <header>
        <h2>Point {point_number}</h2>
        <p>Comparaison des positions localisees, du filtrage statistique et de la ground truth temporelle.</p>
      </header>
      <div class="point-layout">
        <div class="metrics-grid">{metrics}</div>
        <div class="images">
          <figure>
            <img src="{position_img}" alt="Positions vs ground truth point {point_number}">
            <figcaption>Plan ENU: positions trouvees, positions gardees, moyenne des positions gardees et ground truth.</figcaption>
          </figure>
        </div>
      </div>
    </section>
    """


def image_src(image_path: Path) -> str:
    """Retourne une source HTML autonome pour une image PNG."""
    encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded_image}"


def tdoa_slide(point_number: int) -> str:
    if TDOA_RESULTS_DIR is None:
        return ""

    image_path = (
        TDOA_RESULTS_DIR
        / "plots"
        / f"point_{point_number}_tdoa_distributions.png"
    )
    if not image_path.is_file():
        return ""

    image_src_value = image_src(image_path)
    return f"""
    <section class="slide tdoa-slide">
      <header>
        <h2>Point {point_number} - Dispersion des TDOA</h2>
        <p>Repartition par tetraedre et paire d'hydrophones, apres exclusion des valeurs situees a plus de deux ecarts types.</p>
      </header>
      <figure class="tdoa-figure">
        <img src="{image_src_value}" alt="Distributions et ecarts types des TDOA du point {point_number}">
        <figcaption>La ligne rouge indique la moyenne finale. La moyenne, l'ecart type et le nombre de mesures conservees sont indiques sous chaque paire.</figcaption>
      </figure>
    </section>
    """


def direction_slide(point_number: int) -> str:
    image_path = RESULTS_DIR / f"point_{point_number}_direction_map.png"
    if not image_path.is_file():
        return ""
    image_src_value = image_src(image_path)
    return f"""
    <section class="slide direction-slide">
      <header>
        <h2>Point {point_number} - Vecteurs directeurs</h2>
        <p>Directions ENU estimees independamment par chaque tetraedre et appariees aux positions conservees.</p>
      </header>
      <figure class="direction-figure">
        <img src="{image_src_value}" alt="Carte des vecteurs directeurs du point {point_number}">
        <figcaption>Seules les directions valides, issues d'un evenement localise et associees temporellement a une position gardee, sont affichees. Les fleches representent la projection horizontale des vecteurs unitaires.</figcaption>
      </figure>
    </section>
    """


def build_html(summary_df: pd.DataFrame) -> str:
    best_gt = summary_df.loc[summary_df["mean_distance_to_ground_truth_m"].idxmin()]
    worst_gt = summary_df.loc[summary_df["mean_distance_to_ground_truth_m"].idxmax()]
    most_compact = summary_df.loc[summary_df["mean_distance_to_mean_xy_m"].idxmin()]
    least_compact = summary_df.loc[summary_df["mean_distance_to_mean_xy_m"].idxmax()]

    point_sections = []
    for _, row in summary_df.iterrows():
        point_number = int(row["point_number"])
        point_sections.append(point_slide(row))
        current_direction_slide = direction_slide(point_number)
        if current_direction_slide:
            point_sections.append(current_direction_slide)
        current_tdoa_slide = tdoa_slide(point_number)
        if current_tdoa_slide:
            point_sections.append(current_tdoa_slide)
    point_slides = "\n".join(point_sections)

    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Resultats localisation Beluga</title>
  <style>
    :root {{
      --text: #17202a;
      --muted: #59636f;
      --line: #d9dee5;
      --blue: #2454a6;
      --orange: #b75b13;
      --green: #237a4b;
      --bg: #f5f7fa;
      --slide: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--text);
      background: var(--bg);
    }}
    .pdf-button {{
      position: fixed;
      z-index: 1000;
      top: 18px;
      right: 18px;
      padding: 12px 18px;
      border: 0;
      border-radius: 7px;
      color: white;
      background: var(--blue);
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 4px 14px rgba(25, 35, 45, 0.25);
    }}
    .pdf-button:hover {{ background: #183d7d; }}
    .slide {{
      width: 1280px;
      min-height: 720px;
      margin: 24px auto;
      padding: 40px 48px;
      background: var(--slide);
      border: 1px solid var(--line);
      box-shadow: 0 8px 30px rgba(25, 35, 45, 0.12);
      page-break-after: always;
    }}
    h1, h2 {{ margin: 0 0 12px; }}
    h1 {{ font-size: 44px; }}
    h2 {{ font-size: 34px; }}
    p {{ color: var(--muted); font-size: 18px; line-height: 1.35; }}
    .hero {{
      display: grid;
      align-content: center;
      gap: 18px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-top: 28px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-left: 5px solid var(--blue);
      padding: 14px 16px;
      background: #fbfcfe;
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 8px;
    }}
    .metric-value {{
      font-size: 23px;
      font-weight: 700;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 20px;
      font-size: 18px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: right;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); font-size: 14px; text-transform: uppercase; }}
    .point-layout {{
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 24px;
      align-items: start;
    }}
    .metrics-grid {{
      display: grid;
      gap: 10px;
    }}
    .images {{
      display: block;
    }}
    .tdoa-slide, .direction-slide {{
      display: flex;
      flex-direction: column;
    }}
    .tdoa-figure, .direction-figure {{
      flex: 1;
      min-height: 0;
    }}
    .tdoa-slide img, .direction-slide img {{
      height: 570px;
    }}
    figure {{ margin: 0; }}
    img {{
      width: 100%;
      height: 540px;
      object-fit: contain;
      border: 1px solid var(--line);
      background: white;
    }}
    figcaption {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.25;
    }}
    .note {{
      margin-top: 24px;
      padding: 16px 18px;
      border-left: 5px solid var(--orange);
      background: #fff8f2;
      font-size: 18px;
      color: var(--text);
    }}
    @page {{
      size: A4 landscape;
      margin: 0;
    }}
    @media print {{
      body {{ background: white; }}
      .pdf-button {{ display: none; }}
      .slide {{
        margin: 0;
        box-shadow: none;
        border: none;
        width: 297mm;
        height: 210mm;
        min-height: 210mm;
        padding: 10mm 12mm;
        overflow: hidden;
        break-after: page;
        page-break-after: always;
      }}
    }}
  </style>
</head>
<body>
  <button class="pdf-button" type="button" onclick="window.print()">Exporter en PDF</button>
  <section class="slide hero">
    <div>
      <h1>Resultats de localisation 2D</h1>
      <p>Analyse des points {int(summary_df['point_number'].min())} a {int(summary_df['point_number'].max())}: filtrage par ecart type, comparaison a la ground truth et dispersion autour de la moyenne.</p>
    </div>
    <div class="summary-grid">
      {metric_card("Nombre de points", str(len(summary_df)))}
      {metric_card("Meilleur point vs GT", f"Point {int(best_gt['point_number'])} - {fmt(best_gt['mean_distance_to_ground_truth_m'])} m")}
      {metric_card("Point le plus disperse vs GT", f"Point {int(worst_gt['point_number'])} - {fmt(worst_gt['mean_distance_to_ground_truth_m'])} m")}
      {metric_card("Point le plus compact", f"Point {int(most_compact['point_number'])} - {fmt(most_compact['mean_distance_to_mean_xy_m'])} m")}
    </div>
    <div class="note">La distance a la ground truth est calculee point par point avec la ground truth correspondant au timestamp. La distance a la moyenne mesure la dispersion des positions gardees autour de leur position moyenne.</div>
  </section>

  <section class="slide">
    <h2>Synthese globale</h2>
    <p>Les distances moyennes et les ecarts types affiches sont calcules uniquement sur les positions conservees par le filtre statistique.</p>
    {summary_table(summary_df)}
    <div class="note">Le point le moins compact est le point {int(least_compact['point_number'])}, avec une distance moyenne a la moyenne de {fmt(least_compact['mean_distance_to_mean_xy_m'])} m.</div>
  </section>

  {point_slides}
</body>
</html>
"""


def main() -> None:
    global RESULTS_DIR, TDOA_RESULTS_DIR, DIRECTION_RESULTS_DIR, SUMMARY_PATH, OUTPUT_PATH

    args = parse_args()

    if args.position_results is not None:
        resolved_position_results = resolve_input_dir(args.position_results)
        if resolved_position_results is None:
            raise ValueError("--position-results ne peut pas valoir 'none'.")
        RESULTS_DIR = resolved_position_results
    if args.tdoa_results is not None:
        TDOA_RESULTS_DIR = resolve_input_dir(args.tdoa_results)
    if args.direction_results is not None:
        DIRECTION_RESULTS_DIR = resolve_input_dir(args.direction_results)

    SUMMARY_PATH = RESULTS_DIR / "summary.csv"
    OUTPUT_PATH = resolve_output_path(args.output)
    validate_results_dirs()

    if OUTPUT_PATH.exists() and not args.overwrite:
        raise FileExistsError(
            f"{OUTPUT_PATH} existe deja. Choisis un autre --output ou ajoute --overwrite."
        )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(SUMMARY_PATH)
    summary_df.columns = [column.strip() for column in summary_df.columns]
    summary_df = add_kept_stats(summary_df)
    environment = Environment(ENV_PATH, use_h4=True)
    directions_df = read_directions()
    for point_number in summary_df["point_number"].astype(int):
        generate_presentation_enu_plot(point_number, environment)
        generate_direction_map(point_number, environment, directions_df)
    html = build_html(summary_df)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Position stats used: {RESULTS_DIR}")
    if TDOA_RESULTS_DIR is not None:
        print(f"TDOA plots used: {TDOA_RESULTS_DIR}")
    if DIRECTION_RESULTS_DIR is not None:
        print(f"Directions used: {DIRECTION_RESULTS_DIR}")
    print(f"Presentation generated at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
