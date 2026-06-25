from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("test_data2026_all") / "results" / "position_stats"
SUMMARY_PATH = RESULTS_DIR / "summary.csv"
OUTPUT_PATH = RESULTS_DIR / "position_stats_presentation.html"


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

    position_img = f"point_{point_number}_positions_vs_ground_truth.png"
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
            <figcaption>Plan ENU: positions trouvees, positions gardees, moyenne et ground truth.</figcaption>
          </figure>
        </div>
      </div>
    </section>
    """


def build_html(summary_df: pd.DataFrame) -> str:
    best_gt = summary_df.loc[summary_df["mean_distance_to_ground_truth_m"].idxmin()]
    worst_gt = summary_df.loc[summary_df["mean_distance_to_ground_truth_m"].idxmax()]
    most_compact = summary_df.loc[summary_df["mean_distance_to_mean_xy_m"].idxmin()]
    least_compact = summary_df.loc[summary_df["mean_distance_to_mean_xy_m"].idxmax()]

    point_slides = "\n".join(point_slide(row) for _, row in summary_df.iterrows())

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
    @media print {{
      body {{ background: white; }}
      .slide {{
        margin: 0;
        box-shadow: none;
        border: none;
        width: 100%;
        min-height: 100vh;
      }}
    }}
  </style>
</head>
<body>
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
    summary_df = pd.read_csv(SUMMARY_PATH)
    summary_df.columns = [column.strip() for column in summary_df.columns]
    summary_df = add_kept_stats(summary_df)
    html = build_html(summary_df)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Presentation generated at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
