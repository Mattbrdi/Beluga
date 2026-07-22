from __future__ import annotations

import argparse
import base64
import csv
import errno
import io
import json
import math
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import numpy as np
    from scipy import signal as sp_signal
    import dash
    from dash import Input, Output, State, dash_table, dcc, html
    import plotly.graph_objects as go
except ModuleNotFoundError as exc:  # pragma: no cover - message shown at runtime
    raise SystemExit(
        "Dependances manquantes pour l'interface. Installe numpy, scipy, dash et plotly, "
        "par exemple:\n"
        "  pip install numpy scipy dash plotly\n"
    ) from exc

try:
    from .io import load_scene
except ImportError:  # pragma: no cover - useful when launched as a file
    from BSS.Benchmark.io import load_scene


DEFAULT_RESULTS = Path("BSS/Benchmark/results/boat_and_whistle_v2")
DEFAULT_DATASET = Path("BSS/Dataset/generated/boat_and_whistle_v2")


@dataclass(frozen=True)
class AppConfig:
    results_root: Path
    dataset_root: Path | None


@dataclass(frozen=True)
class SignalGroup:
    label: str
    data: np.ndarray
    fs: int
    traces: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interface Dash pour explorer les resultats BSS benchmark."
    )
    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS,
        type=Path,
        help="Dossier racine des resultats benchmark.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        type=Path,
        help="Dossier racine du dataset original. Par defaut, detection automatique.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Adresse d'ecoute Dash.")
    parser.add_argument("--port", default=8050, type=int, help="Port Dash.")
    parser.add_argument("--debug", action="store_true", help="Active le mode debug Dash.")
    return parser.parse_args()


def _as_posixish_path(value: str | Path) -> Path:
    return Path(str(value).replace("\\", "/"))


def _existing_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None

    candidate = _as_posixish_path(value)
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.append(Path.cwd() / candidate)

    for path in candidates:
        if path.exists():
            return path
    return None


def _infer_dataset_root(results_root: Path, cli_dataset: Path | None) -> Path | None:
    if cli_dataset is not None:
        return cli_dataset if cli_dataset.exists() else None

    candidates = [
        DEFAULT_DATASET,
        Path("BSS/Dataset/generated") / results_root.name,
        results_root.parents[2] / "Dataset" / "generated" / results_root.name
        if len(results_root.parents) >= 3
        else Path(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _discover_splits(results_root: Path, dataset_root: Path | None = None) -> list[str]:
    splits = {path.name for path in results_root.iterdir() if path.is_dir()}
    if dataset_root is not None and dataset_root.exists():
        splits.update(path.name for path in dataset_root.iterdir() if path.is_dir())

    summary = _read_summary(results_root / "summary.csv")
    splits.update(row["split"] for row in summary if row.get("split"))
    return sorted(splits)


def _discover_scenes(
    results_root: Path,
    split: str,
    dataset_root: Path | None = None,
) -> list[str]:
    scenes: set[str] = set()

    split_dir = results_root / split
    if split_dir.exists():
        scenes.update(path.name for path in split_dir.iterdir() if path.is_dir())

    if dataset_root is not None:
        manifest = dataset_root / split / "manifest.jsonl"
        if manifest.exists():
            with manifest.open("r", encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        record = json.loads(line)
                        if record.get("id"):
                            scenes.add(record["id"])

    summary = _read_summary(results_root / "summary.csv")
    scenes.update(
        row["scene_id"]
        for row in summary
        if row.get("split") == split and row.get("scene_id")
    )
    return sorted(scenes)


def _discover_algorithms(results_root: Path, split: str, scene_id: str) -> list[str]:
    scene_dir = results_root / split / scene_id
    algorithms = sorted(
        path.name.removesuffix("_metrics.json")
        for path in scene_dir.glob("*_metrics.json")
    )
    if algorithms:
        return algorithms

    summary = _read_summary(results_root / "summary.csv")
    return sorted(
        {
            row["algorithm"]
            for row in summary
            if row.get("split") == split
            and row.get("scene_id") == scene_id
            and row.get("algorithm")
        }
    )


def _metrics_path(results_root: Path, split: str, scene_id: str, algorithm: str) -> Path:
    return results_root / split / scene_id / f"{algorithm}_metrics.json"


def _sources_path(results_root: Path, split: str, scene_id: str, algorithm: str) -> Path:
    return results_root / split / scene_id / f"{algorithm}_sources.npz"


def _sawada_model_path(results_root: Path, split: str, scene_id: str) -> Path:
    return results_root / split / scene_id / "sawada_model.npz"


def _scene_path_from_manifest(dataset_root: Path, split: str, scene_id: str) -> Path | None:
    manifest = dataset_root / split / "manifest.jsonl"
    if not manifest.exists():
        return None

    with manifest.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("id") == scene_id:
                return dataset_root / split / record["path"]
    return None


def _resolve_scene_path(
    dataset_root: Path | None,
    split: str,
    scene_id: str,
    metrics: dict[str, Any],
) -> Path | None:
    scene_path = _existing_path(metrics.get("scene_path"))
    if scene_path is not None:
        return scene_path

    if dataset_root is None:
        return None

    manifest_path = _scene_path_from_manifest(dataset_root, split, scene_id)
    if manifest_path is not None and manifest_path.exists():
        return manifest_path

    numeric_suffix = scene_id.split("_")[-1]
    direct_path = dataset_root / split / f"scene_{numeric_suffix}.npz"
    return direct_path if direct_path.exists() else None


def _npz_array(path: Path, key: str) -> np.ndarray | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as payload:
        if key not in payload:
            return None
        return np.asarray(payload[key]).copy()


def _load_sawada_model(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]).copy() for key in payload.files}


@lru_cache(maxsize=16)
def _load_bundle(
    results_root_text: str,
    dataset_root_text: str,
    split: str,
    scene_id: str,
    algorithm: str,
) -> dict[str, Any]:
    results_root = Path(results_root_text)
    dataset_root = Path(dataset_root_text) if dataset_root_text else None
    metrics = _read_json(_metrics_path(results_root, split, scene_id, algorithm))

    sources_npz_path = _sources_path(results_root, split, scene_id, algorithm)
    estimated_sources = _npz_array(sources_npz_path, "sources")
    fs = int(metrics.get("fs") or _npz_array(sources_npz_path, "fs") or 0)
    sawada_model = (
        _load_sawada_model(_sawada_model_path(results_root, split, scene_id))
        if algorithm == "sawada"
        else {}
    )
 

    scene_path = _resolve_scene_path(dataset_root, split, scene_id, metrics)
    scene_arrays: dict[str, np.ndarray] = {}
    scene_metadata: dict[str, Any] = {}
    if scene_path is not None:
        scene = load_scene(scene_path)
        fs = int(scene.metadata.fs)
        scene_arrays = {
            "sources": scene.sources.data,
            "mixed": scene.mixed.data,
            "clean_mixed": scene.clean_mixed.data,
            "noise": scene.noise.data,
        }
        scene_metadata = {
            "duration": scene.metadata.duration,
            "n_sources": scene.metadata.n_sources,
            "n_mics": scene.metadata.n_mics,
            "max_delay": scene.metadata.max_delay,
            "snr_db": scene.metadata.snr_db,
            "seed": scene.metadata.seed,
            "scene_path": str(scene_path),
        }

    return {
        "metrics": metrics,
        "scene_arrays": scene_arrays,
        "scene_metadata": scene_metadata,
        "estimated_sources": estimated_sources,
        "sawada_model": sawada_model,
        "fs": fs,
    }


def _bundle(config: AppConfig, split: str, scene_id: str, algorithm: str) -> dict[str, Any]:
    return _load_bundle(
        str(config.results_root),
        "" if config.dataset_root is None else str(config.dataset_root),
        split,
        scene_id,
        algorithm,
    )


def _trace_labels(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix} {index + 1}" for index in range(count))


def _signal_groups(bundle: dict[str, Any]) -> dict[str, SignalGroup]:
    fs = int(bundle["fs"])
    groups: dict[str, SignalGroup] = {}

    scene_arrays = bundle["scene_arrays"]
    if "sources" in scene_arrays:
        sources = np.asarray(scene_arrays["sources"])
        groups["sources"] = SignalGroup(
            "Sources originales",
            sources,
            fs,
            _trace_labels("Source", sources.shape[0]),
        )
    if "mixed" in scene_arrays:
        mixed = np.asarray(scene_arrays["mixed"])
        groups["mixed"] = SignalGroup(
            "Melanges bruites",
            mixed,
            fs,
            _trace_labels("Micro", mixed.shape[0]),
        )
    if "clean_mixed" in scene_arrays:
        clean = np.asarray(scene_arrays["clean_mixed"])
        groups["clean_mixed"] = SignalGroup(
            "Melanges propres",
            clean,
            fs,
            _trace_labels("Micro", clean.shape[0]),
        )
    if "noise" in scene_arrays:
        noise = np.asarray(scene_arrays["noise"])
        groups["noise"] = SignalGroup(
            "Bruit",
            noise,
            fs,
            _trace_labels("Micro", noise.shape[0]),
        )

    estimated = bundle["estimated_sources"]
    if estimated is not None and estimated.size:
        estimated = np.asarray(estimated)
        if estimated.ndim == 3:
            n_sources, n_mics, n_samples = estimated.shape
            flat = estimated.reshape(n_sources * n_mics, n_samples)
            labels = tuple(
                f"Est. source {source + 1} - M{mic + 1}"
                for source in range(n_sources)
                for mic in range(n_mics)
            )
        elif estimated.ndim == 2:
            flat = estimated
            labels = _trace_labels("Estimation", estimated.shape[0])
        else:
            flat = estimated.reshape(1, -1)
            labels = ("Estimation",)
        groups["estimated"] = SignalGroup("Sources estimees", flat, fs, labels)

    return groups


def _peak_normalize(data: np.ndarray) -> np.ndarray:
    peak = float(np.nanmax(np.abs(data))) if data.size else 0.0
    if peak <= 0 or not math.isfinite(peak):
        return np.zeros_like(data, dtype=float)
    return np.asarray(data, dtype=float) / peak


def _downsample_for_plot(data: np.ndarray, max_points: int = 7000) -> tuple[np.ndarray, int]:
    if data.shape[-1] <= max_points:
        return data, 1
    step = int(math.ceil(data.shape[-1] / max_points))
    return data[..., ::step], step


def _time_figure(group: SignalGroup) -> go.Figure:
    data = np.asarray(group.data, dtype=float)
    plot_data, step = _downsample_for_plot(data)
    times = np.arange(plot_data.shape[1]) * step / group.fs

    fig = go.Figure()
    for index, label in enumerate(group.traces):
        normalized = _peak_normalize(plot_data[index])
        fig.add_trace(
            go.Scattergl(
                x=times,
                y=normalized + index * 2.4,
                mode="lines",
                name=label,
                line={"width": 1.2},
            )
        )
    fig.update_layout(
        margin={"l": 52, "r": 18, "t": 34, "b": 42},
        title="Signaux temporels",
        xaxis_title="Temps (s)",
        yaxis_title="Amplitude normalisee",
        paper_bgcolor="#f7f7f3",
        plot_bgcolor="#ffffff",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.16},
    )
    return fig


def _spectrogram_figure(
    group: SignalGroup,
    trace_index: int,
    nperseg: int,
    max_frequency: float | None,
    frequency_scale: str = "linear",
    spectrogram_scale: str = "db",
) -> go.Figure:
    signal = np.asarray(group.data[trace_index], dtype=float)
    if signal.size < 8:
        return go.Figure()

    nperseg = nperseg or 2048
    nperseg = max(64, min(int(nperseg), signal.size))
    noverlap = min(nperseg // 2, nperseg - 1)
    freqs, times, magnitude = sp_signal.spectrogram(
        signal,
        fs=group.fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
    )

    values = (
        20 * np.log10(magnitude + 1e-12)
        if spectrogram_scale == "db"
        else magnitude
    )
    if max_frequency is not None and max_frequency > 0:
        mask = freqs <= max_frequency
        freqs = freqs[mask]
        values = values[mask, :]

    yaxis_type = "log" if frequency_scale == "log" else "linear"
    if yaxis_type == "log":
        mask = freqs > 0
        freqs = freqs[mask]
        values = values[mask, :]

    lower_percentile = 5 if spectrogram_scale == "db" else 1
    zmin = float(np.nanpercentile(values, lower_percentile)) if values.size else 0.0
    zmax = float(np.nanpercentile(values, 99)) if values.size else 1.0
    if zmax <= zmin:
        zmax = zmin + 1.0

    fig = go.Figure(
        data=go.Heatmap(
            z=values,
            x=times,
            y=freqs,
            colorscale="Turbo",
            zmin=zmin,
            zmax=zmax,
            colorbar={"title": "dB" if spectrogram_scale == "db" else "mag."},
        )
    )
    fig.update_layout(
        margin={"l": 58, "r": 18, "t": 34, "b": 42},
        title=f"Spectrogramme - {group.traces[trace_index]}",
        xaxis_title="Temps (s)",
        yaxis_title="Frequence (Hz)",
        yaxis_type=yaxis_type,
        paper_bgcolor="#f7f7f3",
        plot_bgcolor="#ffffff",
    )
    return fig


def _sawada_mask_figure(
    sawada_model: dict[str, np.ndarray],
    source_index: int,
    frequency_scale: str = "linear",
    map_kind: str = "mask",
) -> go.Figure:
    masks = np.asarray(sawada_model.get("masks", []), dtype=float)
    posteriors = np.asarray(sawada_model.get("posteriors", []), dtype=float)
    frequencies = np.asarray(sawada_model.get("frequencies", []), dtype=float)
    times = np.asarray(sawada_model.get("times", []), dtype=float)
    fig = go.Figure()

    values_tensor = posteriors if map_kind == "posterior" else masks
    missing_posterior = map_kind == "posterior" and (
        posteriors.ndim != 3 or posteriors.size == 0
    )
    if values_tensor.ndim != 3 or values_tensor.size == 0:
        fig.update_layout(
            title="Masques Sawada indisponibles",
            annotations=[
                {
                    "text": (
                        "Relance le benchmark Sawada pour generer sawada_model.npz."
                        if not missing_posterior
                        else "Relance le benchmark Sawada pour generer les probabilites EM."
                    ),
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                }
            ],
        )
    else:
        source_index = min(max(int(source_index or 0), 0), values_tensor.shape[0] - 1)
        values = values_tensor[source_index]
        yaxis_type = "log" if frequency_scale == "log" else "linear"
        if yaxis_type == "log":
            mask = frequencies > 0
            frequencies = frequencies[mask]
            values = values[mask, :]

        is_posterior = map_kind == "posterior"
        fig.add_trace(
            go.Heatmap(
                z=values,
                x=times,
                y=frequencies,
                colorscale="Viridis"
                if is_posterior
                else [
                    [0.0, "#f7f7f3"],
                    [0.499, "#f7f7f3"],
                    [0.5, "#0f766e"],
                    [1.0, "#0f766e"],
                ],
                zmin=0,
                zmax=1,
                colorbar={"title": "P(source)" if is_posterior else "mask"},
                hovertemplate=(
                    "t=%{x:.4f}s<br>f=%{y:.1f}Hz<br>P=%{z:.3f}<extra></extra>"
                    if is_posterior
                    else "t=%{x:.4f}s<br>f=%{y:.1f}Hz<br>mask=%{z}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title=(
                f"Probabilite EM - Source {source_index + 1}"
                if is_posterior
                else f"Masque Sawada - Source {source_index + 1}"
            ),
            xaxis_title="Temps (s)",
            yaxis_title="Frequence (Hz)",
            yaxis_type=yaxis_type,
        )

    fig.update_layout(
        margin={"l": 58, "r": 18, "t": 34, "b": 42},
        paper_bgcolor="#f7f7f3",
        plot_bgcolor="#ffffff",
    )
    return fig


def _sawada_energy_figure(
    sawada_model: dict[str, np.ndarray],
    frequency_scale: str = "linear",
) -> go.Figure:
    energy = np.asarray(sawada_model.get("tf_energy", []), dtype=float)
    frequencies = np.asarray(sawada_model.get("frequencies", []), dtype=float)
    times = np.asarray(sawada_model.get("times", []), dtype=float)
    fig = go.Figure()

    if energy.ndim != 2 or energy.size == 0:
        fig.update_layout(
            title="Energie Sawada indisponible",
            annotations=[
                {
                    "text": "Relance le benchmark Sawada pour generer l'energie temps-frequence.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                }
            ],
        )
    else:
        values = 10 * np.log10(energy + 1e-20)
        yaxis_type = "log" if frequency_scale == "log" else "linear"
        if yaxis_type == "log":
            mask = frequencies > 0
            frequencies = frequencies[mask]
            values = values[mask, :]

        zmax = float(np.nanpercentile(values, 99)) if values.size else 0.0
        zmin = max(float(np.nanpercentile(values, 5)), zmax - 100.0)
        if zmax <= zmin:
            zmax = zmin + 1.0

        fig.add_trace(
            go.Heatmap(
                z=values,
                x=times,
                y=frequencies,
                colorscale="Magma",
                zmin=zmin,
                zmax=zmax,
                colorbar={"title": "dB"},
                hovertemplate="t=%{x:.4f}s<br>f=%{y:.1f}Hz<br>E=%{z:.1f}dB<extra></extra>",
            )
        )
        fig.update_layout(
            title="Energie temps-frequence du melange",
            xaxis_title="Temps (s)",
            yaxis_title="Frequence (Hz)",
            yaxis_type=yaxis_type,
        )

    fig.update_layout(
        margin={"l": 58, "r": 18, "t": 34, "b": 42},
        paper_bgcolor="#f7f7f3",
        plot_bgcolor="#ffffff",
    )
    return fig


def _wav_data_uri(signal: np.ndarray, fs: int) -> str:
    signal = np.asarray(signal, dtype=float)
    signal = np.nan_to_num(signal)
    target_fs = min(int(fs), 48_000)
    if fs > target_fs:
        divisor = math.gcd(int(fs), target_fs)
        signal = sp_signal.resample_poly(signal, target_fs // divisor, int(fs) // divisor)
        fs = target_fs

    signal = _peak_normalize(signal)
    pcm = np.asarray(np.clip(signal, -1.0, 1.0) * 32767, dtype=np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(fs))
        wav_file.writeframes(pcm.tobytes())

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def _metric_value(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get("metrics", {}).get(key)
    if value is None:
        return "-"
    return f"{float(value):.4g}"


def _overview_children(bundle: dict[str, Any]) -> list[Any]:
    metrics = bundle["metrics"]
    meta = bundle["scene_metadata"]
    cards = [
        ("Statut", metrics.get("status", "-")),
        ("RMSE", f"{_metric_value(metrics, 'rmse_samples')} samples"),
        ("MAE", f"{_metric_value(metrics, 'mae_samples')} samples"),
        ("Runtime", f"{float(metrics.get('runtime_seconds', 0.0)):.3f} s"),
        ("Sources", metrics.get("n_sources", meta.get("n_sources", "-"))),
        ("Micros", metrics.get("n_mics", meta.get("n_mics", "-"))),
        ("Fs", f"{int(bundle['fs'])} Hz" if bundle["fs"] else "-"),
        ("SNR", "-" if meta.get("snr_db") is None else f"{meta['snr_db']} dB"),
        ("Scene", Path(meta.get("scene_path", metrics.get("scene_path", "-"))).name),
    ]
    return [
        html.Div(
            [html.Div(label, className="metric-label"), html.Div(str(value), className="metric-value")],
            className="metric-card",
        )
        for label, value in cards
    ]


def _tdoa_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    labels = metrics.get("pairwise_labels") or []
    true_samples = np.asarray(metrics.get("true_pairwise_tdoas_samples", []), dtype=float)
    estimated_samples = np.asarray(metrics.get("estimated_pairwise_tdoas_samples", []), dtype=float)
    aligned_samples = np.asarray(metrics.get("aligned_pairwise_tdoas_samples", []), dtype=float)
    true_seconds = np.asarray(metrics.get("true_pairwise_tdoas_seconds", []), dtype=float)
    aligned_seconds = np.asarray(metrics.get("aligned_pairwise_tdoas_seconds", []), dtype=float)

    rows: list[dict[str, Any]] = []
    if (
        true_samples.ndim != 2
        or aligned_samples.ndim != 2
        or true_seconds.shape != true_samples.shape
        or aligned_seconds.shape != true_samples.shape
    ):
        return rows

    for source_index in range(true_samples.shape[0]):
        for pair_index, label in enumerate(labels):
            estimated = (
                estimated_samples[source_index, pair_index]
                if estimated_samples.shape == true_samples.shape
                else np.nan
            )
            aligned = aligned_samples[source_index, pair_index]
            truth = true_samples[source_index, pair_index]
            rows.append(
                {
                    "source": f"S{source_index + 1}",
                    "pair": label,
                    "true_samples": round(float(truth), 3),
                    "estimated_samples": round(float(aligned), 3),
                    "raw_estimated_samples": round(float(estimated), 3),
                    "error_samples": round(float(aligned - truth), 3),
                    "true_ms": round(float(true_seconds[source_index, pair_index] * 1000), 4),
                    "estimated_ms": round(float(aligned_seconds[source_index, pair_index] * 1000), 4),
                }
            )
    return rows


def _tdoa_figure(metrics: dict[str, Any]) -> go.Figure:
    labels = metrics.get("pairwise_labels") or []
    true_samples = np.asarray(metrics.get("true_pairwise_tdoas_samples", []), dtype=float)
    aligned_samples = np.asarray(metrics.get("aligned_pairwise_tdoas_samples", []), dtype=float)

    fig = go.Figure()
    if true_samples.ndim == 2 and aligned_samples.shape == true_samples.shape:
        errors = aligned_samples - true_samples
        fig.add_trace(
            go.Heatmap(
                z=errors,
                x=labels,
                y=[f"S{index + 1}" for index in range(errors.shape[0])],
                colorscale="RdBu",
                zmid=0,
                colorbar={"title": "samples"},
            )
        )
    fig.update_layout(
        margin={"l": 52, "r": 18, "t": 34, "b": 42},
        title="Erreur TDOA alignee",
        paper_bgcolor="#f7f7f3",
        plot_bgcolor="#ffffff",
    )
    return fig


def _style() -> str:
    return """
        :root {
            color-scheme: light;
            --bg: #f7f7f3;
            --panel: #ffffff;
            --ink: #202522;
            --muted: #64706b;
            --line: #d9ded8;
            --accent: #0f766e;
            --accent-2: #b45309;
            --bad: #b91c1c;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            background: var(--bg);
            color: var(--ink);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .app-shell { min-height: 100vh; }
        .topbar {
            position: sticky;
            top: 0;
            z-index: 20;
            display: grid;
            grid-template-columns: minmax(220px, 1fr) repeat(5, minmax(140px, 190px));
            gap: 12px;
            align-items: end;
            padding: 14px 18px;
            border-bottom: 1px solid var(--line);
            background: rgba(247, 247, 243, 0.96);
            backdrop-filter: blur(8px);
        }
        .brand { align-self: center; }
        .brand h1 { margin: 0; font-size: 20px; line-height: 1.2; letter-spacing: 0; }
        .brand p { margin: 3px 0 0; color: var(--muted); font-size: 12px; }
        label { display: block; color: var(--muted); font-size: 12px; font-weight: 700; margin-bottom: 4px; }
        .content {
            display: grid;
            grid-template-columns: minmax(0, 1.3fr) minmax(420px, 0.7fr);
            gap: 14px;
            padding: 14px 18px 22px;
        }
        .panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }
        .panel-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            min-height: 42px;
            padding: 10px 12px;
            border-bottom: 1px solid var(--line);
        }
        .panel-title { margin: 0; font-size: 14px; font-weight: 800; }
        .panel-body { padding: 12px; }
        .stack { display: grid; gap: 14px; }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
        }
        .metric-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 10px;
            background: #fbfbf8;
        }
        .metric-label { color: var(--muted); font-size: 11px; font-weight: 800; text-transform: uppercase; }
        .metric-value { margin-top: 4px; font-size: 16px; font-weight: 800; }
        .audio-row {
            display: grid;
            grid-template-columns: 1fr;
            gap: 8px;
        }
        audio { width: 100%; height: 38px; }
        .muted { color: var(--muted); font-size: 12px; }
        .warning {
            color: var(--bad);
            background: #fff1f1;
            border: 1px solid #fecaca;
            border-radius: 8px;
            padding: 10px;
            font-size: 13px;
        }
        @media (max-width: 1100px) {
            .topbar { grid-template-columns: 1fr 1fr; }
            .brand { grid-column: 1 / -1; }
            .content { grid-template-columns: 1fr; }
        }
        @media (max-width: 680px) {
            .topbar { grid-template-columns: 1fr; }
            .content { padding: 10px; }
            .metrics-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        """


def build_app(config: AppConfig) -> dash.Dash:
    splits = _discover_splits(config.results_root, config.dataset_root)
    first_split = splits[0] if splits else ""
    scenes = (
        _discover_scenes(config.results_root, first_split, config.dataset_root)
        if first_split
        else []
    )
    first_scene = scenes[0] if scenes else ""
    algorithms = (
        _discover_algorithms(config.results_root, first_split, first_scene)
        if first_scene
        else []
    )
    first_algorithm = algorithms[0] if algorithms else ""

    app = dash.Dash(__name__)
    app.title = "BSS Benchmark"
    app.index_string = (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "  <head>\n"
        "    {%metas%}\n"
        "    <title>{%title%}</title>\n"
        "    {%favicon%}\n"
        "    {%css%}\n"
        f"    <style>{_style()}</style>\n"
        "  </head>\n"
        "  <body>\n"
        "    {%app_entry%}\n"
        "    <footer>\n"
        "      {%config%}\n"
        "      {%scripts%}\n"
        "      {%renderer%}\n"
        "    </footer>\n"
        "  </body>\n"
        "</html>\n"
    )
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("BSS Benchmark"),
                            html.P(f"Resultats: {config.results_root}"),
                            html.P(
                                "Dataset: "
                                + ("-" if config.dataset_root is None else str(config.dataset_root))
                            ),
                        ],
                        className="brand",
                    ),
                    html.Div(
                        [
                            html.Label("Split"),
                            dcc.Dropdown(
                                id="split-dropdown",
                                options=[{"label": split, "value": split} for split in splits],
                                value=first_split,
                                clearable=False,
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Scene"),
                            dcc.Dropdown(id="scene-dropdown", value=first_scene, clearable=False),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Algorithme"),
                            dcc.Dropdown(
                                id="algorithm-dropdown",
                                value=first_algorithm,
                                clearable=False,
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Famille"),
                            dcc.Dropdown(id="group-dropdown", clearable=False),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Ecoute/Spectro"),
                            dcc.Dropdown(id="trace-dropdown", clearable=False),
                        ]
                    ),
                ],
                className="topbar",
            ),
            dcc.Interval(id="refresh-interval", interval=5000, n_intervals=0),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(id="warning-slot"),
                            html.Div(id="overview", className="metrics-grid"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H2(
                                                        "Signaux",
                                                        className="panel-title",
                                                    )
                                                ],
                                                className="panel-header",
                                            ),
                                            dcc.Graph(
                                                id="time-graph",
                                                config={"displayModeBar": True},
                                                style={"height": "360px"},
                                            ),
                                        ],
                                        className="panel",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H2(
                                                        "Spectrogramme",
                                                        className="panel-title",
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label("NFFT"),
                                                            dcc.Dropdown(
                                                                id="nperseg-dropdown",
                                                                options=[
                                                                    {"label": str(value), "value": value}
                                                                    for value in (512, 1024, 2048, 4096, 8192)
                                                                ],
                                                                value=2048,
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        style={"width": "120px"},
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label("F max"),
                                                            dcc.Input(
                                                                id="max-frequency-input",
                                                                type="number",
                                                                value=12000,
                                                                min=100,
                                                                step=100,
                                                                style={"width": "100%"},
                                                            ),
                                                        ],
                                                        style={"width": "120px"},
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label("Freq."),
                                                            dcc.Dropdown(
                                                                id="frequency-scale-dropdown",
                                                                options=[
                                                                    {"label": "Lineaire", "value": "linear"},
                                                                    {"label": "Log", "value": "log"},
                                                                ],
                                                                value="linear",
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        style={"width": "120px"},
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label("Valeurs"),
                                                            dcc.Dropdown(
                                                                id="spectrogram-scale-dropdown",
                                                                options=[
                                                                    {"label": "dB", "value": "db"},
                                                                    {"label": "Lineaire", "value": "linear"},
                                                                ],
                                                                value="db",
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        style={"width": "130px"},
                                                    ),
                                                ],
                                                className="panel-header",
                                            ),
                                            dcc.Graph(
                                                id="spectrogram-graph",
                                                config={"displayModeBar": True},
                                                style={"height": "390px"},
                                            ),
                                        ],
                                        className="panel",
                                    ),
                                ],
                                className="stack",
                            ),
                        ],
                        className="stack",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [html.H2("Audio", className="panel-title")],
                                        className="panel-header",
                                    ),
                                    html.Div(
                                        [
                                            html.Audio(id="audio-player", controls=True),
                                            html.Div(id="audio-caption", className="muted"),
                                        ],
                                        className="panel-body audio-row",
                                    ),
                                ],
                                className="panel",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H2("Masques Sawada", className="panel-title"),
                                            html.Div(
                                                [
                                                    html.Label("Carte"),
                                                    dcc.Dropdown(
                                                        id="sawada-map-kind-dropdown",
                                                        options=[
                                                            {"label": "Masque", "value": "mask"},
                                                            {"label": "Probabilite EM", "value": "posterior"},
                                                        ],
                                                        value="mask",
                                                        clearable=False,
                                                    ),
                                                ],
                                                style={"width": "150px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("Source"),
                                                    dcc.Dropdown(
                                                        id="mask-source-dropdown",
                                                        value=0,
                                                        clearable=False,
                                                    ),
                                                ],
                                                style={"width": "120px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("Freq."),
                                                    dcc.Dropdown(
                                                        id="mask-frequency-scale-dropdown",
                                                        options=[
                                                            {"label": "Lineaire", "value": "linear"},
                                                            {"label": "Log", "value": "log"},
                                                        ],
                                                        value="linear",
                                                        clearable=False,
                                                    ),
                                                ],
                                                style={"width": "120px"},
                                            ),
                                        ],
                                        className="panel-header",
                                    ),
                                    dcc.Graph(
                                        id="sawada-mask-graph",
                                        config={"displayModeBar": True},
                                        style={"height": "300px"},
                                    ),
                                    html.Div(
                                        id="sawada-mask-caption",
                                        className="muted panel-body",
                                    ),
                                    dcc.Graph(
                                        id="sawada-energy-graph",
                                        config={"displayModeBar": True},
                                        style={"height": "260px"},
                                    ),
                                    html.Div(
                                        id="sawada-energy-caption",
                                        className="muted panel-body",
                                    ),
                                ],
                                className="panel",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [html.H2("TDOA", className="panel-title")],
                                        className="panel-header",
                                    ),
                                    dcc.Graph(
                                        id="tdoa-graph",
                                        config={"displayModeBar": True},
                                        style={"height": "260px"},
                                    ),
                                    html.Div(
                                        dash_table.DataTable(
                                            id="tdoa-table",
                                            columns=[
                                                {"name": "Source", "id": "source"},
                                                {"name": "Paire", "id": "pair"},
                                                {"name": "Vrai (samples)", "id": "true_samples"},
                                                {"name": "Estime (samples)", "id": "estimated_samples"},
                                                {"name": "Erreur", "id": "error_samples"},
                                                {"name": "Estime brut", "id": "raw_estimated_samples"},
                                                {"name": "Vrai (ms)", "id": "true_ms"},
                                                {"name": "Estime (ms)", "id": "estimated_ms"},
                                            ],
                                            data=[],
                                            sort_action="native",
                                            page_size=12,
                                            style_cell={
                                                "fontFamily": "Inter, sans-serif",
                                                "fontSize": "12px",
                                                "textAlign": "right",
                                                "padding": "7px",
                                            },
                                            style_header={
                                                "fontWeight": "800",
                                                "backgroundColor": "#eef2ed",
                                            },
                                            style_data_conditional=[
                                                {
                                                    "if": {"column_id": "source"},
                                                    "textAlign": "left",
                                                },
                                                {
                                                    "if": {"column_id": "pair"},
                                                    "textAlign": "left",
                                                },
                                            ],
                                        ),
                                        className="panel-body",
                                    ),
                                ],
                                className="panel",
                            ),
                        ],
                        className="stack",
                    ),
                ],
                className="content",
            ),
        ],
        className="app-shell",
    )

    @app.callback(
        Output("split-dropdown", "options"),
        Output("split-dropdown", "value"),
        Input("refresh-interval", "n_intervals"),
        State("split-dropdown", "value"),
    )
    def update_splits(
        _: int,
        current_split: str | None,
    ) -> tuple[list[dict[str, str]], str]:
        splits = _discover_splits(config.results_root, config.dataset_root)
        value = current_split if current_split in splits else (splits[0] if splits else "")
        return [{"label": split, "value": split} for split in splits], value

    @app.callback(
        Output("scene-dropdown", "options"),
        Output("scene-dropdown", "value"),
        Input("split-dropdown", "value"),
        State("scene-dropdown", "value"),
    )
    def update_scenes(split: str, current_scene: str | None) -> tuple[list[dict[str, str]], str]:
        scenes = _discover_scenes(config.results_root, split, config.dataset_root)
        value = current_scene if current_scene in scenes else (scenes[0] if scenes else "")
        return [{"label": scene, "value": scene} for scene in scenes], value

    @app.callback(
        Output("algorithm-dropdown", "options"),
        Output("algorithm-dropdown", "value"),
        Input("split-dropdown", "value"),
        Input("scene-dropdown", "value"),
        State("algorithm-dropdown", "value"),
    )
    def update_algorithms(
        split: str, scene_id: str, current_algorithm: str | None
    ) -> tuple[list[dict[str, str]], str]:
        algorithms = _discover_algorithms(config.results_root, split, scene_id)
        value = (
            current_algorithm
            if current_algorithm in algorithms
            else (algorithms[0] if algorithms else "")
        )
        return [{"label": algorithm, "value": algorithm} for algorithm in algorithms], value

    @app.callback(
        Output("group-dropdown", "options"),
        Output("group-dropdown", "value"),
        Input("split-dropdown", "value"),
        Input("scene-dropdown", "value"),
        Input("algorithm-dropdown", "value"),
        State("group-dropdown", "value"),
    )
    def update_groups(
        split: str, scene_id: str, algorithm: str, current_group: str | None
    ) -> tuple[list[dict[str, str]], str]:
        if not split or not scene_id or not algorithm:
            return [], ""
        groups = _signal_groups(_bundle(config, split, scene_id, algorithm))
        keys = list(groups)
        value = current_group if current_group in groups else (keys[0] if keys else "")
        return [{"label": group.label, "value": key} for key, group in groups.items()], value

    @app.callback(
        Output("trace-dropdown", "options"),
        Output("trace-dropdown", "value"),
        Input("split-dropdown", "value"),
        Input("scene-dropdown", "value"),
        Input("algorithm-dropdown", "value"),
        Input("group-dropdown", "value"),
        State("trace-dropdown", "value"),
    )
    def update_traces(
        split: str,
        scene_id: str,
        algorithm: str,
        group_key: str,
        current_trace: int | None,
    ) -> tuple[list[dict[str, int]], int]:
        if not split or not scene_id or not algorithm or not group_key:
            return [], 0
        groups = _signal_groups(_bundle(config, split, scene_id, algorithm))
        if group_key not in groups:
            group_key = next(iter(groups), "")
        if not group_key:
            return [], 0
        group = groups[group_key]
        indexes = list(range(len(group.traces)))
        value = current_trace if current_trace in indexes else 0
        return [
            {"label": label, "value": index}
            for index, label in enumerate(group.traces)
        ], value

    @app.callback(
        Output("mask-source-dropdown", "options"),
        Output("mask-source-dropdown", "value"),
        Input("split-dropdown", "value"),
        Input("scene-dropdown", "value"),
        Input("algorithm-dropdown", "value"),
        State("mask-source-dropdown", "value"),
    )
    def update_mask_sources(
        split: str,
        scene_id: str,
        algorithm: str,
        current_source: int | None,
    ) -> tuple[list[dict[str, int]], int]:
        if not split or not scene_id or algorithm != "sawada":
            return [], 0

        masks = _bundle(config, split, scene_id, algorithm)["sawada_model"].get("masks")
        if masks is None or np.asarray(masks).ndim != 3:
            return [], 0

        source_indexes = list(range(np.asarray(masks).shape[0]))
        value = current_source if current_source in source_indexes else 0
        return [
            {"label": f"Source {source_index + 1}", "value": source_index}
            for source_index in source_indexes
        ], value

    @app.callback(
        Output("sawada-mask-graph", "figure"),
        Output("sawada-mask-caption", "children"),
        Input("split-dropdown", "value"),
        Input("scene-dropdown", "value"),
        Input("algorithm-dropdown", "value"),
        Input("mask-source-dropdown", "value"),
        Input("mask-frequency-scale-dropdown", "value"),
        Input("sawada-map-kind-dropdown", "value"),
    )
    def update_sawada_mask(
        split: str,
        scene_id: str,
        algorithm: str,
        source_index: int,
        frequency_scale: str,
        map_kind: str,
    ) -> tuple[go.Figure, str]:
        if not split or not scene_id or algorithm != "sawada":
            return _sawada_mask_figure({}, 0), "Disponible uniquement pour Sawada."

        model = _bundle(config, split, scene_id, algorithm)["sawada_model"]
        masks = np.asarray(model.get("masks", []))
        if masks.ndim != 3 or masks.size == 0:
            return (
                _sawada_mask_figure({}, 0),
                "Aucun sawada_model.npz trouve pour cette scene. Relance le benchmark Sawada.",
            )

        source_index = min(max(int(source_index or 0), 0), masks.shape[0] - 1)
        if map_kind == "posterior":
            posteriors = np.asarray(model.get("posteriors", []))
            if posteriors.ndim != 3 or posteriors.size == 0:
                return (
                    _sawada_mask_figure(model, source_index, frequency_scale, map_kind),
                    "Probabilites EM absentes. Relance le benchmark Sawada avec cette version.",
                )
            mean_probability = float(np.mean(posteriors[source_index]))
            caption = (
                f"Probabilite EM source {source_index + 1}: "
                f"{posteriors.shape[1]} bins frequences x {posteriors.shape[2]} trames, "
                f"moyenne {mean_probability:.3f}."
            )
        else:
            active_ratio = float(np.mean(masks[source_index]))
            caption = (
                f"Masque source {source_index + 1}: {masks.shape[1]} bins frequences x "
                f"{masks.shape[2]} trames, occupation {active_ratio:.1%}."
            )
        return _sawada_mask_figure(model, source_index, frequency_scale, map_kind), caption

    @app.callback(
        Output("sawada-energy-graph", "figure"),
        Output("sawada-energy-caption", "children"),
        Input("split-dropdown", "value"),
        Input("scene-dropdown", "value"),
        Input("algorithm-dropdown", "value"),
        Input("mask-frequency-scale-dropdown", "value"),
    )
    def update_sawada_energy(
        split: str,
        scene_id: str,
        algorithm: str,
        frequency_scale: str,
    ) -> tuple[go.Figure, str]:
        if not split or not scene_id or algorithm != "sawada":
            return _sawada_energy_figure({}, frequency_scale), "Disponible uniquement pour Sawada."

        model = _bundle(config, split, scene_id, algorithm)["sawada_model"]
        energy = np.asarray(model.get("tf_energy", []), dtype=float)
        if energy.ndim != 2 or energy.size == 0:
            return (
                _sawada_energy_figure(model, frequency_scale),
                "Energie absente. Relance le benchmark Sawada avec cette version.",
            )

        frequency_energy = np.mean(energy, axis=1)
        db = 10 * np.log10(frequency_energy + 1e-20)
        threshold = np.nanpercentile(db, 10)
        quiet_count = int(np.sum(db <= threshold))
        caption = (
            f"Energie brute avant normalisation: {energy.shape[0]} bins frequences x "
            f"{energy.shape[1]} trames. {quiet_count} bins sont dans les 10% les plus faibles."
        )
        return _sawada_energy_figure(model, frequency_scale), caption

    @app.callback(
        Output("overview", "children"),
        Output("warning-slot", "children"),
        Output("time-graph", "figure"),
        Output("spectrogram-graph", "figure"),
        Output("audio-player", "src"),
        Output("audio-caption", "children"),
        Output("tdoa-graph", "figure"),
        Output("tdoa-table", "data"),
        Input("split-dropdown", "value"),
        Input("scene-dropdown", "value"),
        Input("algorithm-dropdown", "value"),
        Input("group-dropdown", "value"),
        Input("trace-dropdown", "value"),
        Input("nperseg-dropdown", "value"),
        Input("max-frequency-input", "value"),
        Input("frequency-scale-dropdown", "value"),
        Input("spectrogram-scale-dropdown", "value"),
    )
    def update_view(
        split: str,
        scene_id: str,
        algorithm: str,
        group_key: str,
        trace_index: int,
        nperseg: int,
        max_frequency: float | None,
        frequency_scale: str,
        spectrogram_scale: str,
    ) -> tuple[Any, Any, go.Figure, go.Figure, str, str, go.Figure, list[dict[str, Any]]]:
        if not split or not scene_id or not algorithm or not group_key:
            empty = go.Figure()
            return [], "", empty, empty, "", "", empty, []

        bundle = _bundle(config, split, scene_id, algorithm)
        groups = _signal_groups(bundle)
        if group_key not in groups:
            group_key = next(iter(groups), "")
        if not group_key:
            empty = go.Figure()
            return _overview_children(bundle), "", empty, empty, "", "", empty, []
        group = groups[group_key]
        trace_index = int(trace_index or 0)
        trace_index = min(max(trace_index, 0), len(group.traces) - 1)
        signal = group.data[trace_index]
        duration = len(signal) / group.fs if group.fs else 0.0

        warning = ""
        if not bundle["scene_arrays"]:
            warning = html.Div(
                "Dataset original non charge: seules les estimations et les TDOA sont disponibles.",
                className="warning",
            )

        audio_caption = (
            f"{group.label} / {group.traces[trace_index]} - "
            f"{duration:.3f} s, {group.fs} Hz"
        )
        return (
            _overview_children(bundle),
            warning,
            _time_figure(group),
            _spectrogram_figure(
                group,
                trace_index,
                nperseg,
                max_frequency,
                frequency_scale,
                spectrogram_scale,
            ),
            _wav_data_uri(signal, group.fs),
            audio_caption,
            _tdoa_figure(bundle["metrics"]),
            _tdoa_rows(bundle["metrics"]),
        )

    return app


def main() -> None:
    args = parse_args()
    results_root = args.results
    if not results_root.exists():
        raise SystemExit(f"Dossier de resultats introuvable: {results_root}")

    dataset_root = _infer_dataset_root(results_root, args.dataset)
    config = AppConfig(results_root=results_root, dataset_root=dataset_root)
    app = build_app(config)
    print(f"Interface BSS Benchmark: http://{args.host}:{args.port}")
    print(f"  resultats: {config.results_root.resolve()}")
    print(
        "  dataset:   "
        + ("-" if config.dataset_root is None else str(config.dataset_root.resolve()))
    )
    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            raise SystemExit(
                f"Le port {args.port} est deja utilise. Arrete l'ancien serveur "
                f"ou relance avec --port {args.port + 1}."
            ) from exc
        raise


if __name__ == "__main__":
    main()
