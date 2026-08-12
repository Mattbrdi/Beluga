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
    from plotly.subplots import make_subplots
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
            "mixing_filters": scene.mixing.filters,
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


def _stft_kwargs_from_metrics(
    metrics: dict[str, Any],
    sawada_model: dict[str, np.ndarray],
    fs: int,
) -> dict[str, Any]:
    raw = metrics.get("parameters", {}).get("stft_parameters", {})
    frequencies = np.asarray(sawada_model.get("frequencies", []), dtype=float)
    inferred_nperseg = 2 * (frequencies.size - 1) if frequencies.size > 1 else 256
    nperseg = int(raw.get("nperseg") or inferred_nperseg)
    nfft_raw = raw.get("nfft")
    noverlap_raw = raw.get("noverlap")
    return {
        "fs": fs,
        "window": raw.get("window", "hann"),
        "nperseg": nperseg,
        "noverlap": None if noverlap_raw is None else int(noverlap_raw),
        "nfft": None if nfft_raw is None else int(nfft_raw),
        "boundary": raw.get("boundary", "zeros"),
        "padded": bool(raw.get("padded", True)),
        "axis": -1,
    }


def _ground_truth_tf_labels(
    bundle: dict[str, Any],
    sawada_model: dict[str, np.ndarray],
    gt_permutation: str = "identity",
) -> tuple[np.ndarray, np.ndarray]:
    scene_sources = np.asarray(bundle.get("scene_arrays", {}).get("sources", []), dtype=float)
    if scene_sources.ndim != 2 or scene_sources.size == 0 or not int(bundle.get("fs", 0)):
        return np.empty((0, 0), dtype=int), np.empty((0, 0), dtype=bool)

    stft_kwargs = _stft_kwargs_from_metrics(
        bundle["metrics"],
        sawada_model,
        int(bundle["fs"]),
    )
    if scene_sources.shape[-1] < stft_kwargs["nperseg"]:
        stft_kwargs["nperseg"] = scene_sources.shape[-1]
        if stft_kwargs["noverlap"] is not None:
            stft_kwargs["noverlap"] = min(stft_kwargs["noverlap"], stft_kwargs["nperseg"] - 1)
    _, _, source_stft = sp_signal.stft(scene_sources, **stft_kwargs)
    source_energy = np.abs(source_stft) ** 2
    if source_energy.ndim != 3 or source_energy.size == 0:
        return np.empty((0, 0), dtype=int), np.empty((0, 0), dtype=bool)

    labels = np.argmax(source_energy, axis=0).astype(int)
    max_energy = np.max(source_energy, axis=0)
    valid_threshold = float(np.nanmax(max_energy)) * 1e-12 if max_energy.size else 0.0
    valid = max_energy > valid_threshold

    n_sources = source_energy.shape[0]
    permutation = np.arange(n_sources)
    if gt_permutation == "swap" and n_sources >= 2:
        permutation[0], permutation[1] = permutation[1], permutation[0]
    labels = permutation[labels]
    return labels, valid


def _sawada_gt_correctness(
    bundle: dict[str, Any],
    sawada_model: dict[str, np.ndarray],
    gt_permutation: str = "identity",
) -> np.ndarray:
    masks = np.asarray(sawada_model.get("masks", []), dtype=float)
    if masks.ndim != 3 or masks.size == 0:
        return np.empty((0, 0), dtype=float)

    gt_labels, gt_valid = _ground_truth_tf_labels(bundle, sawada_model, gt_permutation)
    if gt_labels.ndim != 2 or gt_labels.size == 0:
        return np.empty((0, 0), dtype=float)

    n_freqs = min(masks.shape[1], gt_labels.shape[0])
    n_times = min(masks.shape[2], gt_labels.shape[1])
    predicted = np.argmax(masks[:, :n_freqs, :n_times], axis=0)
    assigned = np.max(masks[:, :n_freqs, :n_times], axis=0) > 0.5
    valid = assigned & gt_valid[:n_freqs, :n_times]

    correctness = np.zeros((n_freqs, n_times), dtype=float)
    correctness[valid & (predicted == gt_labels[:n_freqs, :n_times])] = 1.0
    correctness[valid & (predicted != gt_labels[:n_freqs, :n_times])] = -1.0
    return correctness


def _normalized_mixture_stft_for_gt(
    bundle: dict[str, Any],
    sawada_model: dict[str, np.ndarray],
) -> np.ndarray:
    vector_key = (
        "bin_vectors_unwhitened"
        if np.asarray(sawada_model.get("bin_vectors_unwhitened", [])).ndim == 3
        else "bin_vectors"
    )
    bin_vectors = np.asarray(sawada_model.get(vector_key, []))
    if bin_vectors.ndim == 3 and bin_vectors.size:
        return bin_vectors

    mixed = np.asarray(bundle.get("scene_arrays", {}).get("mixed", []), dtype=float)
    if mixed.ndim != 2 or mixed.size == 0 or not int(bundle.get("fs", 0)):
        return np.empty((0, 0, 0), dtype=complex)

    stft_kwargs = _stft_kwargs_from_metrics(
        bundle["metrics"],
        sawada_model,
        int(bundle["fs"]),
    )
    if mixed.shape[-1] < stft_kwargs["nperseg"]:
        stft_kwargs["nperseg"] = mixed.shape[-1]
        if stft_kwargs["noverlap"] is not None:
            stft_kwargs["noverlap"] = min(stft_kwargs["noverlap"], stft_kwargs["nperseg"] - 1)
    _, _, mixture_stft = sp_signal.stft(mixed, **stft_kwargs)
    norms = np.linalg.norm(mixture_stft, axis=0, keepdims=True)
    return mixture_stft / (norms + 1e-12)


def _normalized_source_image_stfts_for_gt(
    bundle: dict[str, Any],
    sawada_model: dict[str, np.ndarray],
    gt_permutation: str = "identity",
) -> np.ndarray:
    sources = np.asarray(bundle.get("scene_arrays", {}).get("sources", []), dtype=float)
    mixing_filters = np.asarray(
        bundle.get("scene_arrays", {}).get("mixing_filters", []),
        dtype=float,
    )
    if (
        sources.ndim != 2
        or sources.size == 0
        or mixing_filters.ndim != 3
        or mixing_filters.size == 0
        or not int(bundle.get("fs", 0))
    ):
        return np.empty((0, 0, 0, 0), dtype=complex)

    n_mics, n_sources_from_filters, _ = mixing_filters.shape
    n_sources = min(sources.shape[0], n_sources_from_filters)
    if n_sources <= 0:
        return np.empty((0, 0, 0, 0), dtype=complex)

    source_images = np.zeros((n_sources, n_mics, sources.shape[1]), dtype=float)
    for source_index in range(n_sources):
        for mic_index in range(n_mics):
            source_images[source_index, mic_index] = sp_signal.convolve(
                sources[source_index],
                mixing_filters[mic_index, source_index],
                mode="same",
                method="auto",
            )

    if gt_permutation == "swap" and n_sources >= 2:
        source_images = source_images.copy()
        source_images[[0, 1]] = source_images[[1, 0]]

    stft_kwargs = _stft_kwargs_from_metrics(
        bundle["metrics"],
        sawada_model,
        int(bundle["fs"]),
    )
    if source_images.shape[-1] < stft_kwargs["nperseg"]:
        stft_kwargs["nperseg"] = source_images.shape[-1]
        if stft_kwargs["noverlap"] is not None:
            stft_kwargs["noverlap"] = min(stft_kwargs["noverlap"], stft_kwargs["nperseg"] - 1)
    _, _, source_stfts = sp_signal.stft(source_images, **stft_kwargs)
    norms = np.linalg.norm(source_stfts, axis=1, keepdims=True)
    return source_stfts / (norms + 1e-12)


def _ground_truth_centroids(
    bundle: dict[str, Any],
    sawada_model: dict[str, np.ndarray],
    gt_permutation: str = "identity",
    gt_centroid_mode: str = "mixture_masked",
) -> np.ndarray:
    gt_centroid_mode = gt_centroid_mode or "mixture_masked"
    source_vectors = (
        _normalized_source_image_stfts_for_gt(bundle, sawada_model, gt_permutation)
        if gt_centroid_mode == "source_direct"
        else np.empty((0, 0, 0, 0), dtype=complex)
    )
    use_source_vectors = source_vectors.ndim == 4 and source_vectors.size
    bin_vectors = _normalized_mixture_stft_for_gt(bundle, sawada_model)
    gt_labels, gt_valid = _ground_truth_tf_labels(bundle, sawada_model, gt_permutation)
    if gt_labels.ndim != 2 or gt_valid.ndim != 2:
        return np.empty((0, 0, 0), dtype=complex)

    if use_source_vectors:
        n_mics = source_vectors.shape[1]
        n_freqs = min(source_vectors.shape[2], gt_labels.shape[0], gt_valid.shape[0])
        n_times = min(source_vectors.shape[3], gt_labels.shape[1], gt_valid.shape[1])
    else:
        if bin_vectors.ndim != 3 or bin_vectors.size == 0:
            return np.empty((0, 0, 0), dtype=complex)
        n_mics = bin_vectors.shape[0]
        n_freqs = min(bin_vectors.shape[1], gt_labels.shape[0], gt_valid.shape[0])
        n_times = min(bin_vectors.shape[2], gt_labels.shape[1], gt_valid.shape[1])
    if n_freqs == 0 or n_times == 0:
        return np.empty((0, 0, 0), dtype=complex)

    n_sources = int(np.nanmax(gt_labels[:n_freqs, :n_times])) + 1 if gt_labels.size else 0
    if use_source_vectors:
        n_sources = max(n_sources, source_vectors.shape[0])
    masks = np.asarray(sawada_model.get("masks", []), dtype=float)
    if masks.ndim == 3 and masks.shape[0] > n_sources:
        n_sources = masks.shape[0]
    if n_sources <= 0:
        return np.empty((0, 0, 0), dtype=complex)

    centroids = np.full((n_freqs, n_mics, n_sources), np.nan + 1j * np.nan, dtype=complex)
    for frequency_index in range(n_freqs):
        valid_frames = gt_valid[frequency_index, :n_times]
        for source_index in range(n_sources):
            if use_source_vectors:
                if source_index >= source_vectors.shape[0]:
                    continue
                X_f = source_vectors[source_index, :, frequency_index, :n_times]
            else:
                X_f = bin_vectors[:, frequency_index, :n_times]
            gamma = valid_frames & (gt_labels[frequency_index, :n_times] == source_index)
            if not np.any(gamma):
                continue
            weighted_X = X_f[:, gamma]
            R_i = weighted_X @ weighted_X.conj().T
            try:
                _, eigenvectors = np.linalg.eigh(R_i)
            except np.linalg.LinAlgError:
                continue
            centroid = eigenvectors[:, -1]
            centroids[frequency_index, :, source_index] = centroid / (
                np.linalg.norm(centroid) + 1e-12
            )
    return centroids


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


def _time_figure(group: SignalGroup, trace_index: int | None = None) -> go.Figure:
    data = np.asarray(group.data, dtype=float)
    title = "Signaux temporels"
    if trace_index is not None and data.ndim == 2 and data.size:
        trace_index = min(max(int(trace_index), 0), data.shape[0] - 1)
        data = data[trace_index:trace_index + 1]
        labels = (group.traces[trace_index],)
        title = f"Signal temporel - {group.traces[trace_index]}"
    else:
        labels = group.traces
    plot_data, step = _downsample_for_plot(data)
    times = np.arange(plot_data.shape[1]) * step / group.fs

    fig = go.Figure()
    for index, label in enumerate(labels):
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
        title=title,
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
    stft_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    signal = np.asarray(group.data[trace_index], dtype=float)
    if signal.size < 8:
        return go.Figure()

    if stft_kwargs is None:
        nperseg = nperseg or 2048
        nperseg = max(64, min(int(nperseg), signal.size))
        noverlap = min(nperseg // 2, nperseg - 1)
        stft_kwargs = {
            "fs": group.fs,
            "window": "hann",
            "nperseg": nperseg,
            "noverlap": noverlap,
            "nfft": None,
            "boundary": "zeros",
            "padded": True,
            "axis": -1,
        }
        stft_label = f"STFT manuelle {nperseg}"
    else:
        stft_kwargs = dict(stft_kwargs)
        stft_kwargs["fs"] = group.fs
        nperseg = max(
            8,
            min(int(stft_kwargs.get("nperseg") or nperseg or 2048), signal.size),
        )
        stft_kwargs["nperseg"] = nperseg
        if stft_kwargs.get("noverlap") is not None:
            stft_kwargs["noverlap"] = min(int(stft_kwargs["noverlap"]), nperseg - 1)
        stft_label = f"STFT benchmark {nperseg}"

    freqs, times, stft_values = sp_signal.stft(
        signal,
        **stft_kwargs,
    )
    energy = np.abs(stft_values) ** 2

    values = (
        10 * np.log10(energy + 1e-20)
        if spectrogram_scale == "db"
        else energy
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

    zmax = float(np.nanpercentile(values, 99)) if values.size else 1.0
    if spectrogram_scale == "db":
        zmin = max(
            float(np.nanpercentile(values, 5)) if values.size else zmax - 100.0,
            zmax - 100.0,
        )
    else:
        zmin = float(np.nanpercentile(values, 1)) if values.size else 0.0
    if zmax <= zmin:
        zmax = zmin + 1.0

    fig = go.Figure(
        data=go.Heatmap(
            z=values,
            x=times,
            y=freqs,
            colorscale="Magma",
            zmin=zmin,
            zmax=zmax,
            colorbar={"title": "dB" if spectrogram_scale == "db" else "energie"},
        )
    )
    fig.update_layout(
        margin={"l": 58, "r": 18, "t": 34, "b": 42},
        title=f"Spectrogramme energie - {group.traces[trace_index]} ({stft_label})",
        xaxis_title="Temps (s)",
        yaxis_title="Frequence (Hz)",
        yaxis_type=yaxis_type,
        paper_bgcolor="#f7f7f3",
        plot_bgcolor="#ffffff",
    )
    return fig


def _energy_db_limits(values: np.ndarray) -> tuple[float, float]:
    finite_values = np.asarray(values)[np.isfinite(values)]
    if finite_values.size == 0:
        return -100.0, 0.0

    zmax = float(np.nanpercentile(finite_values, 99))
    zmin = max(float(np.nanpercentile(finite_values, 5)), zmax - 100.0)
    if zmax <= zmin:
        zmax = zmin + 1.0
    return zmin, zmax


def _energy_linear_limits(values: np.ndarray) -> tuple[float, float]:
    finite_values = np.asarray(values)[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0, 1.0

    zmin = float(np.nanpercentile(finite_values, 1))
    zmax = float(np.nanpercentile(finite_values, 99))
    if zmax <= zmin:
        zmax = zmin + 1.0
    return zmin, zmax


def _source_index_for_masked_spectrogram(
    bundle: dict[str, Any],
    group_key: str,
    trace_index: int,
) -> int | None:
    if group_key == "estimated":
        estimated = bundle.get("estimated_sources")
        if estimated is None:
            return None
        estimated = np.asarray(estimated)
        if estimated.ndim == 3 and estimated.shape[1] > 0:
            source_index = trace_index // estimated.shape[1]
            return source_index if 0 <= source_index < estimated.shape[0] else None
        if estimated.ndim == 2:
            return trace_index if 0 <= trace_index < estimated.shape[0] else None

    return None


def _masked_mixture_spectrogram_figure(
    sawada_model: dict[str, np.ndarray],
    source_index: int,
    max_frequency: float | None,
    frequency_scale: str = "linear",
    spectrogram_scale: str = "db",
    title_suffix: str = "",
) -> go.Figure | None:
    masks = np.asarray(sawada_model.get("masks", []), dtype=float)
    tf_energy = np.asarray(sawada_model.get("tf_energy", []), dtype=float)
    frequencies = np.asarray(sawada_model.get("frequencies", []), dtype=float)
    times = np.asarray(sawada_model.get("times", []), dtype=float)
    if masks.ndim != 3 or tf_energy.ndim != 2 or masks.size == 0 or tf_energy.size == 0:
        return None

    source_index = min(max(int(source_index), 0), masks.shape[0] - 1)
    n_freqs = min(masks.shape[1], tf_energy.shape[0])
    n_times = min(masks.shape[2], tf_energy.shape[1])
    energy = masks[source_index, :n_freqs, :n_times] * tf_energy[:n_freqs, :n_times]
    scale_energy = tf_energy[:n_freqs, :n_times]
    frequencies = (
        frequencies[:n_freqs]
        if frequencies.size >= n_freqs
        else np.arange(n_freqs)
    )
    times = times[:n_times] if times.size >= n_times else np.arange(n_times)

    if max_frequency is not None and max_frequency > 0:
        mask = frequencies <= max_frequency
        frequencies = frequencies[mask]
        energy = energy[mask, :]
        scale_energy = scale_energy[mask, :]

    yaxis_type = "log" if frequency_scale == "log" else "linear"
    if yaxis_type == "log":
        mask = frequencies > 0
        frequencies = frequencies[mask]
        energy = energy[mask, :]
        scale_energy = scale_energy[mask, :]

    if spectrogram_scale == "db":
        values = 10 * np.log10(energy + 1e-20)
        zmin, zmax = _energy_db_limits(10 * np.log10(scale_energy + 1e-20))
        colorbar_title = "dB"
    else:
        values = energy
        zmin, zmax = _energy_linear_limits(scale_energy)
        colorbar_title = "energie"

    fig = go.Figure(
        data=go.Heatmap(
            z=values,
            x=times,
            y=frequencies,
            colorscale="Magma",
            zmin=zmin,
            zmax=zmax,
            colorbar={"title": colorbar_title},
            hovertemplate=(
                "t=%{x:.4f}s<br>f=%{y:.1f}Hz<br>E masquee=%{z:.3g}"
                "<extra></extra>"
            ),
        )
    )
    suffix = f" - {title_suffix}" if title_suffix else ""
    fig.update_layout(
        margin={"l": 58, "r": 18, "t": 34, "b": 42},
        title=f"STFT(melange) x masque S{source_index + 1}{suffix}",
        xaxis_title="Temps (s)",
        yaxis_title="Frequence (Hz)",
        yaxis_type=yaxis_type,
        paper_bgcolor="#f7f7f3",
        plot_bgcolor="#ffffff",
    )
    return fig


def _comparison_spectrogram_figure(
    bundle: dict[str, Any],
    group_key: str,
    group: SignalGroup,
    trace_index: int,
    nperseg: int,
    max_frequency: float | None,
    frequency_scale: str,
    spectrogram_scale: str,
    stft_kwargs: dict[str, Any] | None,
) -> go.Figure:
    source_index = _source_index_for_masked_spectrogram(bundle, group_key, trace_index)
    if source_index is not None:
        figure = _masked_mixture_spectrogram_figure(
            bundle["sawada_model"],
            source_index,
            max_frequency,
            frequency_scale,
            spectrogram_scale,
            group.traces[trace_index],
        )
        if figure is not None:
            return figure

    return _spectrogram_figure(
        group,
        trace_index,
        nperseg,
        max_frequency,
        frequency_scale,
        spectrogram_scale,
        stft_kwargs,
    )


def _sawada_mask_figure(
    sawada_model: dict[str, np.ndarray],
    source_index: int,
    frequency_scale: str = "linear",
    map_kind: str = "mask",
    gt_correctness: np.ndarray | None = None,
) -> go.Figure:
    masks = np.asarray(sawada_model.get("masks", []), dtype=float)
    posteriors = np.asarray(sawada_model.get("posteriors", []), dtype=float)
    active_tf_mask = np.asarray(sawada_model.get("active_tf_mask", []), dtype=float)
    tf_energy = np.asarray(sawada_model.get("tf_energy", []), dtype=float)
    frequencies = np.asarray(sawada_model.get("frequencies", []), dtype=float)
    times = np.asarray(sawada_model.get("times", []), dtype=float)
    fig = go.Figure()

    if map_kind == "posterior":
        values_tensor = posteriors
    elif map_kind == "masked_energy":
        if masks.ndim == 3 and tf_energy.ndim == 2:
            n_freqs = min(masks.shape[1], tf_energy.shape[0])
            n_times = min(masks.shape[2], tf_energy.shape[1])
            values_tensor = np.zeros_like(masks, dtype=float)
            values_tensor[:, :n_freqs, :n_times] = (
                masks[:, :n_freqs, :n_times] * tf_energy[np.newaxis, :n_freqs, :n_times]
            )
        else:
            values_tensor = np.empty((0, 0, 0))
    elif map_kind == "active":
        values_tensor = active_tf_mask[np.newaxis, :, :] if active_tf_mask.ndim == 2 else active_tf_mask
    elif map_kind == "gt_error":
        gt_values = np.asarray(gt_correctness if gt_correctness is not None else [])
        values_tensor = gt_values[np.newaxis, :, :] if gt_values.ndim == 2 else gt_values
    else:
        values_tensor = masks
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
        frequencies = (
            frequencies[: values.shape[0]]
            if frequencies.size >= values.shape[0]
            else np.arange(values.shape[0])
        )
        times = (
            times[: values.shape[1]]
            if times.size >= values.shape[1]
            else np.arange(values.shape[1])
        )
        is_posterior = map_kind == "posterior"
        is_active = map_kind == "active"
        is_gt_error = map_kind == "gt_error"
        is_masked_energy = map_kind == "masked_energy"
        tf_energy_for_scale = (
            tf_energy[: values.shape[0], : values.shape[1]]
            if is_masked_energy and tf_energy.ndim == 2
            else np.empty((0, 0))
        )
        yaxis_type = "log" if frequency_scale == "log" else "linear"
        if yaxis_type == "log":
            mask = frequencies > 0
            frequencies = frequencies[mask]
            values = values[mask, :]
            if tf_energy_for_scale.size:
                tf_energy_for_scale = tf_energy_for_scale[mask, :]
        display_values = (
            10 * np.log10(values + 1e-20)
            if is_masked_energy
            else values
        )
        if is_masked_energy and display_values.size:
            zmin, zmax = _energy_db_limits(10 * np.log10(tf_energy_for_scale + 1e-20))
        else:
            zmin = -1 if is_gt_error else 0
            zmax = 1
        fig.add_trace(
            go.Heatmap(
                z=display_values,
                x=times,
                y=frequencies,
                colorscale="Viridis"
                if is_posterior
                else "Magma"
                if is_masked_energy
                else [
                    [0.0, "#dc2626"],
                    [0.49, "#dc2626"],
                    [0.5, "#e5e7eb"],
                    [0.51, "#16a34a"],
                    [1.0, "#16a34a"],
                ]
                if is_gt_error
                else [
                    [0.0, "#f7f7f3"],
                    [0.499, "#f7f7f3"],
                    [0.5, "#0f766e"],
                    [1.0, "#0f766e"],
                ],
                zmin=zmin,
                zmax=zmax,
                colorbar={
                    "title": (
                        "P(source)"
                        if is_posterior
                        else "E dB"
                        if is_masked_energy
                        else "GT"
                        if is_gt_error
                        else "active"
                        if is_active
                        else "mask"
                    ),
                    **(
                        {"tickvals": [-1, 0, 1], "ticktext": ["faux", "ignore", "correct"]}
                        if is_gt_error
                        else {}
                    ),
                },
                hovertemplate=(
                    "t=%{x:.4f}s<br>f=%{y:.1f}Hz<br>P=%{z:.3f}<extra></extra>"
                    if is_posterior
                    else "t=%{x:.4f}s<br>f=%{y:.1f}Hz<br>E masque=%{z:.1f}dB<extra></extra>"
                    if is_masked_energy
                    else "t=%{x:.4f}s<br>f=%{y:.1f}Hz<br>GT=%{z}<extra></extra>"
                    if is_gt_error
                    else "t=%{x:.4f}s<br>f=%{y:.1f}Hz<br>active=%{z}<extra></extra>"
                    if is_active
                    else "t=%{x:.4f}s<br>f=%{y:.1f}Hz<br>mask=%{z}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title=(
                f"Probabilite EM - Source {source_index + 1}"
                if is_posterior
                else f"Energie masquee - Source {source_index + 1}"
                if is_masked_energy
                else "Correct vs GT"
                if is_gt_error
                else "Bins actifs EM"
                if is_active
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

        zmin, zmax = _energy_db_limits(values)

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


def _sawada_complex_plane_figure(
    sawada_model: dict[str, np.ndarray],
    frequency_index: int | None,
    vector_space: str = "whitened",
    coordinate_mode: str = "relative",
    color_mode: str = "source",
    show_rejected_bins: bool = False,
    gt_correctness: np.ndarray | None = None,
) -> go.Figure:
    vector_key = "bin_vectors_unwhitened" if vector_space == "unwhitened" else "bin_vectors"
    centroid_key = "centroids_unwhitened" if vector_space == "unwhitened" else "centroids"
    bin_vectors = np.asarray(sawada_model.get(vector_key, []))
    masks = np.asarray(sawada_model.get("masks", []), dtype=float)
    active_tf_mask = np.asarray(sawada_model.get("active_tf_mask", []), dtype=bool)
    active_clusters = np.asarray(sawada_model.get("active_clusters", []), dtype=bool)
    centroids = np.asarray(sawada_model.get(centroid_key, []))
    frequencies = np.asarray(sawada_model.get("frequencies", []), dtype=float)
    times = np.asarray(sawada_model.get("times", []), dtype=float)

    if (
        bin_vectors.ndim != 3
        or masks.ndim != 3
        or centroids.ndim != 3
        or bin_vectors.size == 0
        or masks.size == 0
        or centroids.size == 0
    ):
        fig = go.Figure()
        fig.update_layout(
            title="Plans complexes Sawada indisponibles",
            annotations=[
                {
                    "text": (
                        "Relance le benchmark Sawada pour sauvegarder les vecteurs complexes "
                        "dans ce repere."
                    ),
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                }
            ],
            margin={"l": 58, "r": 18, "t": 42, "b": 42},
            paper_bgcolor="#f7f7f3",
            plot_bgcolor="#ffffff",
        )
        return fig

    n_mics, n_freqs, n_times = bin_vectors.shape
    n_sources = masks.shape[0]
    frequency_index = min(max(int(frequency_index or 0), 0), n_freqs - 1)
    frequency_label = (
        f"{float(frequencies[frequency_index]):.1f} Hz"
        if frequencies.size > frequency_index
        else f"bin {frequency_index}"
    )

    rows = int(math.ceil(n_mics / 2))
    fig = make_subplots(
        rows=rows,
        cols=2,
        subplot_titles=[f"Micro {mic_index + 1}" for mic_index in range(n_mics)],
    )
    source_colors = ["#dc2626", "#2563eb", "#16a34a", "#7c3aed", "#0891b2", "#db2777"]
    source_masks = masks[:, frequency_index, :]
    used_by_em = (
        active_tf_mask[frequency_index]
        if active_tf_mask.ndim == 2 and active_tf_mask.shape[0] > frequency_index
        else np.max(source_masks, axis=0) > 0.5
    )
    rejected_by_energy = ~used_by_em
    source_active = (
        active_clusters[frequency_index]
        if active_clusters.ndim == 2 and active_clusters.shape[0] > frequency_index
        else np.ones(n_sources, dtype=bool)
    )
    assigned = np.argmax(source_masks, axis=0)
    assigned_strength = np.max(source_masks, axis=0)
    has_assignment = assigned_strength > 0.5
    time_values = times if times.size == n_times else np.arange(n_times)
    bin_values = bin_vectors[:, frequency_index, :].copy()
    centroid_values = centroids[frequency_index].copy()
    reference_text = "composantes directes"
    if coordinate_mode == "relative":
        reference = bin_values[0]
        bin_values = bin_values * np.conj(reference)[np.newaxis, :]

        centroid_reference = centroid_values[0]
        centroid_values = centroid_values * np.conj(centroid_reference)[np.newaxis, :]
        reference_text = "M*conj(M1)"
    gt_values = np.asarray(gt_correctness if gt_correctness is not None else [])
    has_gt = color_mode == "gt" and gt_values.ndim == 2 and gt_values.shape[0] > frequency_index
    correctness = np.zeros(n_times, dtype=float)
    if has_gt:
        common_times = min(n_times, gt_values.shape[1])
        correctness[:common_times] = gt_values[frequency_index, :common_times]

    scale_selector = np.ones(n_times, dtype=bool)
    if not show_rejected_bins:
        scale_selector = scale_selector & used_by_em
    finite_bin_values = bin_values[:, scale_selector].reshape(-1)
    finite_values = np.concatenate(
        [
            finite_bin_values[np.isfinite(finite_bin_values)],
            centroid_values.reshape(-1)[np.isfinite(centroid_values.reshape(-1))],
        ]
    )
    if finite_values.size:
        axis_values = np.concatenate([np.real(finite_values), np.imag(finite_values)])
        axis_half_width = float(np.nanmax(np.abs(axis_values)))
        if axis_half_width <= 1e-9:
            axis_half_width = 1.0
        axis_half_width *= 1.08
        common_axis_range = [-axis_half_width, axis_half_width]
    else:
        common_axis_range = [-1.0, 1.0]

    for mic_index in range(n_mics):
        row = mic_index // 2 + 1
        col = mic_index % 2 + 1
        values = bin_values[mic_index]
        component_label = (
            f"M{mic_index + 1}*conj(M1)"
            if coordinate_mode == "relative"
            else f"Micro {mic_index + 1}"
        )

        if color_mode == "gt":
            classes = [
                ("Correct", used_by_em & (correctness > 0), "#16a34a"),
                ("Faux", used_by_em & (correctness < 0), "#dc2626"),
            ]
        else:
            classes = [
                (
                    f"Source {source_index + 1}",
                    used_by_em & has_assignment & (assigned == source_index),
                    source_colors[source_index % len(source_colors)],
                )
                for source_index in range(n_sources)
            ]

        for label, selector, color in classes:
            if not np.any(selector):
                continue
            fig.add_trace(
                go.Scattergl(
                    x=np.real(values[selector]),
                    y=np.imag(values[selector]),
                    mode="markers",
                    name=label,
                    legendgroup=label,
                    showlegend=mic_index == 0,
                    marker={"size": 6, "color": color, "opacity": 0.72},
                    customdata=time_values[selector],
                    hovertemplate=(
                        f"{label}<br>"
                        "t=%{customdata:.4f}s<br>Re=%{x:.4f}<br>Im=%{y:.4f}"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        rejected_selector = rejected_by_energy
        if show_rejected_bins and np.any(rejected_selector):
            fig.add_trace(
                go.Scattergl(
                    x=np.real(values[rejected_selector]),
                    y=np.imag(values[rejected_selector]),
                    mode="markers",
                    name="Hors EM",
                    legendgroup="rejected",
                    showlegend=mic_index == 0,
                    marker={
                        "size": 5,
                        "color": "#facc15",
                        "opacity": 0.62,
                        "symbol": "circle-open",
                        "line": {"width": 1.2, "color": "#a16207"},
                    },
                    customdata=time_values[rejected_selector],
                    hovertemplate=(
                        "Hors EM<br>t=%{customdata:.4f}s<br>Re=%{x:.4f}"
                        "<br>Im=%{y:.4f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        for source_index in range(min(n_sources, centroids.shape[2])):
            if source_index < source_active.size and not source_active[source_index]:
                continue
            centroid = centroid_values[mic_index, source_index]
            fig.add_trace(
                go.Scatter(
                    x=[float(np.real(centroid))],
                    y=[float(np.imag(centroid))],
                    mode="markers+text",
                    name=f"Centroide S{source_index + 1}",
                    legendgroup=f"centroid-{source_index}",
                    showlegend=mic_index == 0,
                    text=[f"C{source_index + 1}"],
                    textposition="top center",
                    marker={
                        "size": 12,
                        "color": "#f59e0b",
                        "symbol": "diamond",
                        "line": {"width": 1.4, "color": "#111827"},
                    },
                    hovertemplate=(
                        f"Centroide S{source_index + 1}<br>"
                        "Re=%{x:.4f}<br>Im=%{y:.4f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(
            title_text="Re",
            range=common_axis_range,
            zeroline=True,
            zerolinecolor="#111827",
            zerolinewidth=1.4,
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text="Im",
            range=common_axis_range,
            zeroline=True,
            zerolinecolor="#111827",
            zerolinewidth=1.4,
            scaleanchor=f"x{mic_index + 1}" if mic_index > 0 else "x",
            scaleratio=1,
            row=row,
            col=col,
        )
        fig.layout.annotations[mic_index].text = component_label

    fig.update_layout(
        title=(
            f"Vecteurs complexes des bins - {frequency_label} "
            f"({'non blanchi' if vector_space == 'unwhitened' else 'blanchi'}, "
            f"{reference_text})"
        ),
        margin={"l": 58, "r": 18, "t": 64, "b": 42},
        paper_bgcolor="#f7f7f3",
        plot_bgcolor="#ffffff",
        legend={"orientation": "h", "y": -0.12},
    )
    return fig


def _sawada_centroid_phase_figure(
    sawada_model: dict[str, np.ndarray],
    color_mode: str = "frequency",
    reference_mode: str = "relative",
    angle_mode: str = "delta_previous",
    frequency_min: float | None = None,
    frequency_max: float | None = None,
) -> go.Figure:
    centroids = np.asarray(sawada_model.get("centroids_unwhitened", []))
    if centroids.ndim != 3 or centroids.size == 0:
        centroids = np.asarray(sawada_model.get("centroids", []))
    frequencies = np.asarray(sawada_model.get("frequencies", []), dtype=float)

    if centroids.ndim != 3 or centroids.size == 0:
        fig = go.Figure()
        fig.update_layout(
            title="Phases des centroides indisponibles",
            annotations=[
                {
                    "text": "Relance le benchmark Sawada pour sauvegarder les centroides.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                }
            ],
            margin={"l": 58, "r": 18, "t": 42, "b": 42},
            paper_bgcolor="#f7f7f3",
            plot_bgcolor="#ffffff",
        )
        return fig

    if frequencies.size == centroids.shape[1] and frequencies.size != centroids.shape[0]:
        centroids = np.moveaxis(centroids, 1, 0)

    n_freqs, n_mics, n_sources = centroids.shape
    if frequencies.size < n_freqs:
        frequencies = np.arange(n_freqs, dtype=float)
    else:
        frequencies = frequencies[:n_freqs]

    reference_mode = reference_mode or "relative"
    angle_mode = angle_mode or "delta_previous"
    frequency_min = None if frequency_min is None else float(frequency_min)
    frequency_max = None if frequency_max is None else float(frequency_max)
    if (
        frequency_min is not None
        and frequency_max is not None
        and frequency_min > frequency_max
    ):
        frequency_min, frequency_max = frequency_max, frequency_min
    use_relative_phase = reference_mode == "relative"
    use_delta_previous = angle_mode == "delta_previous"

    if use_relative_phase:
        reference = centroids[:, 0, :]
        valid_reference = np.ones((n_freqs, n_sources), dtype=bool)
        phase_input = centroids * np.conj(reference)[:, np.newaxis, :]
    else:
        valid_reference = np.ones((n_freqs, n_sources), dtype=bool)
        phase_input = centroids

    frequency_band = np.isfinite(frequencies)
    if frequency_min is not None:
        frequency_band &= frequencies >= frequency_min
    if frequency_max is not None:
        frequency_band &= frequencies <= frequency_max
    phase_values = np.angle(phase_input)
    if use_delta_previous:
        theta_values_all = np.full_like(phase_values, np.nan, dtype=float)
        if n_freqs > 1:
            theta_values_all[1:] = np.angle(
                np.exp(1j * (phase_values[1:] - phase_values[:-1]))
            )
        valid_angle_frequency = np.arange(n_freqs) > 0
    else:
        theta_values_all = phase_values
        valid_angle_frequency = np.ones(n_freqs, dtype=bool)
    theta_unit = "rad"
    finite_mask = (
        np.isfinite(theta_values_all)
        & valid_reference[:, np.newaxis, :]
        & valid_angle_frequency[:, np.newaxis, np.newaxis]
        & frequency_band[:, np.newaxis, np.newaxis]
    )

    rows = int(math.ceil(n_mics / 2))
    fig = make_subplots(
        rows=rows,
        cols=2,
        subplot_titles=[
            f"M{mic_index + 1}*conj(M1)" if use_relative_phase else f"M{mic_index + 1}"
            for mic_index in range(n_mics)
        ],
    )
    source_colors = ["#dc2626", "#2563eb", "#16a34a", "#7c3aed", "#0891b2", "#db2777"]
    source_frequency_colorscales = [
        [[0.0, "#fee2e2"], [0.55, "#ef4444"], [1.0, "#7f1d1d"]],
        [[0.0, "#dbeafe"], [0.55, "#3b82f6"], [1.0, "#1e3a8a"]],
        [[0.0, "#dcfce7"], [0.55, "#22c55e"], [1.0, "#14532d"]],
        [[0.0, "#ede9fe"], [0.55, "#8b5cf6"], [1.0, "#4c1d95"]],
        [[0.0, "#ccfbf1"], [0.55, "#14b8a6"], [1.0, "#134e4a"]],
        [[0.0, "#fce7f3"], [0.55, "#ec4899"], [1.0, "#831843"]],
    ]
    color_mode = color_mode or "frequency"

    unit_x = np.cos(theta_values_all)
    unit_y = np.sin(theta_values_all)
    circle_angles = np.linspace(0.0, 2.0 * np.pi, 241)
    circle_x = np.cos(circle_angles)
    circle_y = np.sin(circle_angles)
    axis_range = [-1.08, 1.08]

    for mic_index in range(n_mics):
        row = mic_index // 2 + 1
        col = mic_index % 2 + 1
        theta_values = theta_values_all[:, mic_index, :]
        valid_values = finite_mask[:, mic_index, :]

        fig.add_trace(
            go.Scatter(
                x=circle_x,
                y=circle_y,
                mode="lines",
                name="Cercle unite",
                legendgroup="unit-circle",
                showlegend=mic_index == 0,
                line={"color": "#9ca3af", "width": 1.2},
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        if color_mode == "source":
            for source_index in range(n_sources):
                selector = valid_values[:, source_index]
                if not np.any(selector):
                    continue
                fig.add_trace(
                    go.Scattergl(
                        x=unit_x[selector, mic_index, source_index],
                        y=unit_y[selector, mic_index, source_index],
                        mode="markers",
                        name=f"Source {source_index + 1}",
                        legendgroup=f"centroid-phase-source-{source_index}",
                        showlegend=mic_index == 0,
                        marker={
                            "size": 7,
                            "color": source_colors[source_index % len(source_colors)],
                            "opacity": 0.78,
                        },
                        customdata=np.stack(
                            [
                                frequencies[selector],
                                theta_values[selector, source_index],
                            ],
                            axis=1,
                        ),
                        hovertemplate=(
                            f"Source {source_index + 1}<br>"
                            "f=%{customdata[0]:.1f}Hz<br>"
                            f"theta=%{{customdata[1]:.4e}} {theta_unit}<br>"
                            "x=%{x:.4f}<br>y=%{y:.4f}"
                            "<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
        elif color_mode == "source_frequency":
            for source_index in range(n_sources):
                selector = valid_values[:, source_index]
                if not np.any(selector):
                    continue
                fig.add_trace(
                    go.Scattergl(
                        x=unit_x[selector, mic_index, source_index],
                        y=unit_y[selector, mic_index, source_index],
                        mode="markers",
                        name=f"Source {source_index + 1}",
                        legendgroup=f"centroid-phase-source-frequency-{source_index}",
                        showlegend=mic_index == 0,
                        marker={
                            "size": 7,
                            "color": frequencies[selector],
                            "colorscale": source_frequency_colorscales[
                                source_index % len(source_frequency_colorscales)
                            ],
                            "showscale": mic_index == 0,
                            "opacity": 0.8,
                            "colorbar": {
                                "title": f"S{source_index + 1} Hz",
                                "x": 1.02 + 0.08 * source_index,
                                "len": 0.78,
                            },
                        },
                        customdata=np.stack(
                            [
                                frequencies[selector],
                                theta_values[selector, source_index],
                            ],
                            axis=1,
                        ),
                        hovertemplate=(
                            f"Source {source_index + 1}<br>"
                            "f=%{customdata[0]:.1f}Hz<br>"
                            f"theta=%{{customdata[1]:.4e}} {theta_unit}<br>"
                            "x=%{x:.4f}<br>y=%{y:.4f}"
                            "<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
        else:
            x_values = np.repeat(frequencies[:, np.newaxis], n_sources, axis=1)
            source_indices = np.repeat(
                np.arange(1, n_sources + 1)[np.newaxis, :],
                n_freqs,
                axis=0,
            )
            selector = valid_values
            if np.any(selector):
                customdata = np.stack(
                    [
                        source_indices[selector],
                        x_values[selector],
                        theta_values[selector],
                    ],
                    axis=1,
                )
                fig.add_trace(
                    go.Scattergl(
                        x=unit_x[:, mic_index, :][selector],
                        y=unit_y[:, mic_index, :][selector],
                        mode="markers",
                        name="Centroide",
                        showlegend=False,
                        marker={
                            "size": 7,
                            "color": x_values[selector],
                            "coloraxis": "coloraxis",
                            "opacity": 0.76,
                        },
                        customdata=customdata,
                        hovertemplate=(
                            "Source %{customdata[0]:.0f}<br>"
                            "f=%{customdata[1]:.1f}Hz<br>"
                            f"theta=%{{customdata[2]:.4e}} {theta_unit}<br>"
                            "x=%{x:.4f}<br>y=%{y:.4f}"
                            "<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )

        fig.update_xaxes(
            title_text="cos(theta)",
            range=axis_range,
            zeroline=True,
            zerolinecolor="#111827",
            zerolinewidth=1.4,
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text="sin(theta)",
            range=axis_range,
            zeroline=True,
            zerolinecolor="#111827",
            zerolinewidth=1.4,
            scaleanchor=f"x{mic_index + 1}" if mic_index > 0 else "x",
            scaleratio=1,
            row=row,
            col=col,
        )

    fig.update_layout(
        title=(
            "Centroides Sawada sur cercle unite - "
            f"{'Cm*conj(M1)' if use_relative_phase else 'Cm'}, "
            f"{'delta arg precedent' if use_delta_previous else 'arg'}"
        ),
        margin={"l": 58, "r": 18, "t": 64, "b": 42},
        paper_bgcolor="#f7f7f3",
        plot_bgcolor="#ffffff",
        legend={"orientation": "h", "y": -0.12},
    )
    if color_mode == "frequency":
        fig.update_layout(
            coloraxis={
                "colorscale": "Turbo",
                "colorbar": {"title": "Hz"},
            }
        )
    return fig


def _sawada_centroid_argument_frequency_figure(
    sawada_model: dict[str, np.ndarray],
    frequency_min: float | None = None,
    frequency_max: float | None = None,
    gt_centroids: np.ndarray | None = None,
) -> go.Figure:
    centroids = np.asarray(sawada_model.get("centroids_unwhitened", []))
    if centroids.ndim != 3 or centroids.size == 0:
        centroids = np.asarray(sawada_model.get("centroids", []))
    relative_phases = np.asarray(
        sawada_model.get("source_assignment_relative_phases", []),
        dtype=float,
    )
    frequencies = np.asarray(sawada_model.get("frequencies", []), dtype=float)

    if centroids.ndim == 3 and centroids.size:
        if frequencies.size == centroids.shape[1] and frequencies.size != centroids.shape[0]:
            centroids = np.moveaxis(centroids, 1, 0)

        n_freqs, n_mics, n_sources = centroids.shape
        relative_centroids = centroids * np.conj(centroids[:, 0, :])[:, np.newaxis, :]
        argument_values = np.angle(relative_centroids)
    elif relative_phases.ndim == 3 and relative_phases.size:
        n_freqs, n_sources, n_mics = relative_phases.shape
        argument_values = np.moveaxis(relative_phases, 2, 1)
    else:
        fig = go.Figure()
        fig.update_layout(
            title="Arguments des centroides indisponibles",
            annotations=[
                {
                    "text": "Relance le benchmark Sawada pour sauvegarder les centroides ou les phases RANSAC.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                }
            ],
            margin={"l": 58, "r": 18, "t": 42, "b": 42},
            paper_bgcolor="#f7f7f3",
            plot_bgcolor="#ffffff",
        )
        return fig

    if frequencies.size < n_freqs:
        frequencies = np.arange(n_freqs, dtype=float)
    else:
        frequencies = frequencies[:n_freqs]
    assignment_labels = np.asarray(
        sawada_model.get("source_assignment_labels", []),
        dtype=int,
    )
    use_final_assignment = assignment_labels.shape == (n_freqs, n_sources)
    if use_final_assignment and np.any(assignment_labels >= 0):
        n_final_sources = max(n_sources, int(np.max(assignment_labels)) + 1)
    else:
        n_final_sources = n_sources

    frequency_min = None if frequency_min is None else float(frequency_min)
    frequency_max = None if frequency_max is None else float(frequency_max)
    if (
        frequency_min is not None
        and frequency_max is not None
        and frequency_min > frequency_max
    ):
        frequency_min, frequency_max = frequency_max, frequency_min

    frequency_band = np.isfinite(frequencies)
    if frequency_min is not None:
        frequency_band &= frequencies >= frequency_min
    if frequency_max is not None:
        frequency_band &= frequencies <= frequency_max

    gt_centroids = np.asarray(gt_centroids if gt_centroids is not None else [])
    if gt_centroids.ndim == 3 and gt_centroids.size:
        if gt_centroids.shape[1] == n_freqs and gt_centroids.shape[0] != n_freqs:
            gt_centroids = np.moveaxis(gt_centroids, 1, 0)
        gt_n_freqs = min(gt_centroids.shape[0], n_freqs)
        gt_n_mics = min(gt_centroids.shape[1], n_mics)
        gt_n_sources = min(gt_centroids.shape[2], n_sources)
        gt_relative = gt_centroids[:gt_n_freqs, :gt_n_mics, :gt_n_sources] * np.conj(
            gt_centroids[:gt_n_freqs, 0, :gt_n_sources]
        )[:, np.newaxis, :]
        gt_argument_values = np.angle(gt_relative)
    else:
        gt_n_freqs = gt_n_mics = gt_n_sources = 0
        gt_argument_values = np.empty((0, 0, 0), dtype=float)

    rows = int(math.ceil(n_mics / 2))
    fig = make_subplots(
        rows=rows,
        cols=2,
        subplot_titles=[f"M{mic_index + 1}*conj(M1)" for mic_index in range(n_mics)],
    )
    source_colors = ["#dc2626", "#2563eb", "#16a34a", "#7c3aed", "#0891b2", "#db2777"]
    repeated_frequencies = np.repeat(frequencies[:, np.newaxis], n_sources, axis=1)
    repeated_clusters = np.repeat(
        np.arange(n_sources)[np.newaxis, :],
        n_freqs,
        axis=0,
    )

    for mic_index in range(n_mics):
        row = mic_index // 2 + 1
        col = mic_index % 2 + 1
        if use_final_assignment:
            finite_values = np.isfinite(argument_values[:, mic_index, :])
            unassigned = (
                frequency_band[:, np.newaxis]
                & finite_values
                & (assignment_labels < 0)
            )
            if np.any(unassigned):
                customdata = np.stack(
                    [
                        repeated_clusters[unassigned] + 1,
                        repeated_frequencies[unassigned],
                    ],
                    axis=1,
                )
                fig.add_trace(
                    go.Scattergl(
                        x=repeated_frequencies[unassigned],
                        y=argument_values[:, mic_index, :][unassigned],
                        mode="markers",
                        name="Non attribue RANSAC",
                        legendgroup="centroid-argument-unassigned",
                        showlegend=mic_index == 0,
                        marker={"size": 5, "color": "#9ca3af", "opacity": 0.45},
                        customdata=customdata,
                        hovertemplate=(
                            "Non attribue RANSAC<br>"
                            "cluster EM=%{customdata[0]:.0f}<br>"
                            "f=%{customdata[1]:.1f}Hz<br>"
                            "arg=%{y:.4f} rad"
                            "<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
            for source_index in range(n_final_sources):
                selector = (
                    frequency_band[:, np.newaxis]
                    & finite_values
                    & (assignment_labels == source_index)
                )
                if not np.any(selector):
                    continue
                customdata = np.stack(
                    [
                        repeated_clusters[selector] + 1,
                        repeated_frequencies[selector],
                    ],
                    axis=1,
                )
                fig.add_trace(
                    go.Scattergl(
                        x=repeated_frequencies[selector],
                        y=argument_values[:, mic_index, :][selector],
                        mode="markers",
                        name=f"Sawada final S{source_index + 1}",
                        legendgroup=f"centroid-argument-final-source-{source_index}",
                        showlegend=mic_index == 0,
                        marker={
                            "size": 5.5,
                            "color": source_colors[source_index % len(source_colors)],
                            "opacity": 0.82,
                        },
                        customdata=customdata,
                        hovertemplate=(
                            f"Source finale RANSAC S{source_index + 1}<br>"
                            "cluster EM=%{customdata[0]:.0f}<br>"
                            "f=%{customdata[1]:.1f}Hz<br>"
                            "arg=%{y:.4f} rad"
                            "<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
        else:
            for source_index in range(n_sources):
                selector = frequency_band & np.isfinite(argument_values[:, mic_index, source_index])
                if not np.any(selector):
                    continue
                fig.add_trace(
                    go.Scattergl(
                        x=frequencies[selector],
                        y=argument_values[selector, mic_index, source_index],
                        mode="lines+markers",
                        name=f"Cluster EM {source_index + 1}",
                        legendgroup=f"centroid-argument-source-{source_index}",
                        showlegend=mic_index == 0,
                        line={
                            "color": source_colors[source_index % len(source_colors)],
                            "width": 1.5,
                        },
                        marker={
                            "size": 5,
                            "color": source_colors[source_index % len(source_colors)],
                            "opacity": 0.82,
                        },
                        hovertemplate=(
                            f"Cluster EM {source_index + 1}<br>"
                            "f=%{x:.1f}Hz<br>arg=%{y:.4f} rad"
                            "<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
        for source_index in range(gt_n_sources):
            if (
                mic_index < gt_n_mics
                and gt_n_freqs > 0
            ):
                gt_selector = (
                    frequency_band[:gt_n_freqs]
                    & np.isfinite(gt_argument_values[:, mic_index, source_index])
                )
                if np.any(gt_selector):
                    fig.add_trace(
                        go.Scattergl(
                            x=frequencies[:gt_n_freqs][gt_selector],
                            y=gt_argument_values[:, mic_index, source_index][gt_selector],
                            mode="lines+markers",
                            name=f"GT S{source_index + 1}",
                            legendgroup=f"centroid-argument-gt-source-{source_index}",
                            showlegend=mic_index == 0,
                            line={
                                "color": source_colors[source_index % len(source_colors)],
                                "width": 1.4,
                                "dash": "dash",
                            },
                            marker={
                                "size": 5,
                                "color": "#ffffff",
                                "line": {
                                    "width": 1.4,
                                    "color": source_colors[source_index % len(source_colors)],
                                },
                            },
                            hovertemplate=(
                                f"GT S{source_index + 1}<br>"
                                "f=%{x:.1f}Hz<br>arg=%{y:.4f} rad"
                                "<extra></extra>"
                            ),
                        ),
                        row=row,
                        col=col,
                    )

        fig.update_xaxes(
            title_text="Frequence (Hz)",
            zeroline=True,
            zerolinecolor="#111827",
            zerolinewidth=1.2,
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text="arg (rad)",
            range=[-np.pi, np.pi],
            zeroline=True,
            zerolinecolor="#111827",
            zerolinewidth=1.2,
            row=row,
            col=col,
        )

    fig.update_layout(
        title=(
            "Arguments des centroides relatifs en fonction de la frequence - "
            f"{'sources finales RANSAC' if use_final_assignment else 'clusters EM locaux'}"
        ),
        margin={"l": 58, "r": 18, "t": 64, "b": 42},
        paper_bgcolor="#f7f7f3",
        plot_bgcolor="#ffffff",
        legend={"orientation": "h", "y": -0.12},
    )
    return fig


def _wrap_phase(values: np.ndarray | float) -> np.ndarray | float:
    return ((np.asarray(values) + np.pi) % (2.0 * np.pi)) - np.pi


def _phase_line_with_breaks(
    x_values: np.ndarray,
    y_values: np.ndarray,
    jump_threshold: float = np.pi,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[valid]
    y_values = y_values[valid]
    if x_values.size < 2:
        return x_values, y_values

    x_out: list[float] = [float(x_values[0])]
    y_out: list[float] = [float(y_values[0])]
    jumps = np.abs(np.diff(y_values)) > float(jump_threshold)
    for index in range(1, x_values.size):
        if bool(jumps[index - 1]):
            x_out.append(float("nan"))
            y_out.append(float("nan"))
        x_out.append(float(x_values[index]))
        y_out.append(float(y_values[index]))
    return np.asarray(x_out), np.asarray(y_out)


def _sawada_ransac_figure(
    sawada_model: dict[str, np.ndarray],
    source_index: int = 0,
    color_mode: str = "assignment",
) -> go.Figure:
    relative_phases = np.asarray(
        sawada_model.get("source_assignment_relative_phases", []),
        dtype=float,
    )
    labels = np.asarray(sawada_model.get("source_assignment_labels", []), dtype=int)
    selected_labels = np.asarray(
        sawada_model.get("source_assignment_selected_labels", []),
        dtype=int,
    )
    distances = np.asarray(
        sawada_model.get("source_assignment_distances", []),
        dtype=float,
    )
    slopes = np.asarray(sawada_model.get("source_assignment_slopes", []), dtype=float)
    intercepts = np.asarray(
        sawada_model.get("source_assignment_intercepts", []),
        dtype=float,
    )
    frequencies = np.asarray(sawada_model.get("frequencies", []), dtype=float)
    frequency_inliers = np.asarray(
        sawada_model.get("source_assignment_frequency_inliers", []),
        dtype=bool,
    )
    selected_centroids = np.asarray(
        sawada_model.get("source_assignment_selected_centroids", []),
        dtype=int,
    )

    fig = go.Figure()
    if (
        relative_phases.ndim != 3
        or relative_phases.size == 0
        or labels.shape != relative_phases.shape[:2]
        or slopes.ndim != 2
        or intercepts.shape != slopes.shape
    ):
        fig.update_layout(
            title="RANSAC circulaire indisponible",
            annotations=[
                {
                    "text": "Relance le benchmark Sawada pour sauvegarder les champs source_assignment_*.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                }
            ],
            margin={"l": 58, "r": 18, "t": 42, "b": 42},
            paper_bgcolor="#f7f7f3",
            plot_bgcolor="#ffffff",
        )
        return fig

    n_freqs, n_centroids, n_components = relative_phases.shape
    n_sources = slopes.shape[0]
    if frequencies.size < n_freqs:
        frequencies = np.arange(n_freqs, dtype=float)
    else:
        frequencies = frequencies[:n_freqs]
    source_index = min(max(int(source_index or 0), 0), max(n_sources - 1, 0))
    color_mode = color_mode or "assignment"

    available_points = labels >= 0
    if selected_labels.shape == labels.shape:
        available_points |= selected_labels >= 0
    if distances.ndim == 3 and distances.shape[1:] == labels.shape:
        available_points |= np.any(np.isfinite(distances), axis=0)
    elif distances.shape == labels.shape:
        available_points |= np.isfinite(distances)
    finite_any_component = np.any(np.isfinite(relative_phases), axis=2)
    active_points_base = finite_any_component & available_points
    if not np.any(active_points_base):
        active_points_base = finite_any_component
    active_frequencies = np.any(active_points_base, axis=1)
    repeated_frequencies = np.repeat(frequencies[:, np.newaxis], n_centroids, axis=1)
    source_colors = ["#dc2626", "#2563eb", "#16a34a", "#7c3aed", "#0891b2", "#db2777"]

    finite_frequencies = np.isfinite(frequencies)
    t_ref = float(np.nanmean(frequencies[finite_frequencies])) if np.any(finite_frequencies) else 0.0
    display_frequencies = finite_frequencies & active_frequencies
    if not np.any(display_frequencies):
        display_frequencies = finite_frequencies

    active_frequency_values = frequencies[display_frequencies]
    x_range = None
    if active_frequency_values.size:
        x_min = float(np.nanmin(active_frequency_values))
        x_max = float(np.nanmax(active_frequency_values))
        padding = max((x_max - x_min) * 0.025, 1.0)
        x_range = [x_min - padding, x_max + padding]

    rows = int(math.ceil(n_components / 2))
    cols = 2 if n_components > 1 else 1
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[
            f"M{component_index + 1}*conj(M1)"
            for component_index in range(n_components)
        ],
    )

    for component_index in range(n_components):
        row = component_index // 2 + 1
        col = component_index % 2 + 1
        component_label = f"M{component_index + 1}*conj(M1)"
        phase_values = relative_phases[:, :, component_index]
        finite_points = np.isfinite(phase_values) & np.isfinite(frequencies[:, np.newaxis])
        active_points = finite_points & available_points
        if not np.any(active_points):
            active_points = finite_points

        if color_mode == "distance":
            if distances.ndim == 3 and distances.shape[1:] == labels.shape:
                distance_values = (
                    distances[source_index]
                    if source_index < distances.shape[0]
                    else np.nanmin(distances, axis=0)
                )
            elif distances.shape == labels.shape:
                distance_values = distances
            else:
                distance_values = np.full(labels.shape, np.nan, dtype=float)
            finite = active_points & np.isfinite(distance_values)
            fig.add_trace(
                go.Scattergl(
                    x=repeated_frequencies[finite],
                    y=phase_values[finite],
                    mode="markers",
                    name=f"Distance a S{source_index + 1}",
                    legendgroup=f"ransac-distance-source-{source_index}",
                    showlegend=component_index == 0,
                    marker={
                        "size": 5,
                        "opacity": 0.78,
                        "color": distance_values[finite],
                        "colorscale": "Viridis",
                        "showscale": component_index == 0,
                        "colorbar": {"title": "dist"},
                    },
                    hovertemplate=(
                        f"{component_label}<br>"
                        "f=%{x:.1f} Hz<br>"
                        "phase=%{y:.4f} rad<br>"
                        "dist=%{marker.color:.4f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
        else:
            unassigned = active_points & (labels < 0)
            if np.any(unassigned):
                fig.add_trace(
                    go.Scattergl(
                        x=repeated_frequencies[unassigned],
                        y=phase_values[unassigned],
                        mode="markers",
                        name="Non attribue",
                        legendgroup="ransac-unassigned",
                        showlegend=component_index == 0,
                        marker={"size": 4.5, "color": "#9ca3af", "opacity": 0.34},
                        hovertemplate=(
                            f"{component_label}<br>"
                            "f=%{x:.1f} Hz<br>phase=%{y:.4f} rad<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
            for label_index in range(n_sources):
                selector = active_points & (labels == label_index)
                if not np.any(selector):
                    continue
                fig.add_trace(
                    go.Scattergl(
                        x=repeated_frequencies[selector],
                        y=phase_values[selector],
                        mode="markers",
                        name=f"Attribue S{label_index + 1}",
                        legendgroup=f"ransac-assigned-source-{label_index}",
                        showlegend=component_index == 0,
                        marker={
                            "size": 5.5 if label_index == source_index else 4.5,
                            "color": source_colors[label_index % len(source_colors)],
                            "opacity": 0.82 if label_index == source_index else 0.42,
                        },
                        hovertemplate=(
                            f"{component_label}<br>S{label_index + 1}<br>"
                            "f=%{x:.1f} Hz<br>phase=%{y:.4f} rad<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )

        for label_index in range(n_sources):
            if component_index >= slopes.shape[1]:
                continue
            prediction = _wrap_phase(
                intercepts[label_index, component_index]
                + (frequencies - t_ref) * slopes[label_index, component_index]
            )
            line_x, line_y = _phase_line_with_breaks(
                frequencies[display_frequencies],
                prediction[display_frequencies],
            )
            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode="lines",
                    name=f"RANSAC S{label_index + 1}",
                    legendgroup=f"ransac-line-source-{label_index}",
                    showlegend=component_index == 0,
                    line={
                        "color": source_colors[label_index % len(source_colors)],
                        "width": 3.0 if label_index == source_index else 1.5,
                        "dash": "solid" if label_index == source_index else "dash",
                    },
                    opacity=0.96 if label_index == source_index else 0.58,
                    hovertemplate=(
                        f"{component_label}<br>Modele S{label_index + 1}<br>"
                        "f=%{x:.1f} Hz<br>phase=%{y:.4f} rad<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        if (
            frequency_inliers.ndim == 2
            and selected_centroids.ndim == 2
            and source_index < frequency_inliers.shape[0]
            and source_index < selected_centroids.shape[0]
        ):
            n_selected_freqs = min(n_freqs, frequency_inliers.shape[1], selected_centroids.shape[1])
            selected = selected_centroids[source_index, :n_selected_freqs]
            inliers = frequency_inliers[source_index, :n_selected_freqs]
            valid_selected = (
                inliers
                & (selected >= 0)
                & (selected < n_centroids)
                & np.isfinite(frequencies[:n_selected_freqs])
            )
            selected_freq_indexes = np.flatnonzero(valid_selected)
            if selected_freq_indexes.size:
                selected_centroid_indexes = selected[selected_freq_indexes]
                fig.add_trace(
                    go.Scattergl(
                        x=frequencies[selected_freq_indexes],
                        y=phase_values[selected_freq_indexes, selected_centroid_indexes],
                        mode="markers",
                        name=f"Inliers RANSAC S{source_index + 1}",
                        legendgroup=f"ransac-inliers-source-{source_index}",
                        showlegend=component_index == 0,
                        marker={
                            "symbol": "diamond-open",
                            "size": 9,
                            "color": "#111827",
                            "line": {"width": 2.0, "color": "#f59e0b"},
                        },
                        hovertemplate=(
                            f"{component_label}<br>Inlier selectionne<br>"
                            "f=%{x:.1f} Hz<br>phase=%{y:.4f} rad<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )

        fig.update_xaxes(
            title_text="Frequence (Hz)",
            zeroline=True,
            zerolinecolor="#111827",
            zerolinewidth=1.2,
            range=x_range,
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text="arg (rad)",
            range=[-np.pi, np.pi],
            zeroline=True,
            zerolinecolor="#111827",
            zerolinewidth=1.2,
            row=row,
            col=col,
        )

    fig.update_layout(
        title=f"RANSAC circulaire - Source {source_index + 1}, toutes composantes",
        margin={"l": 58, "r": 18, "t": 64, "b": 48},
        paper_bgcolor="#f7f7f3",
        plot_bgcolor="#ffffff",
        legend={"orientation": "h", "y": -0.14, "groupclick": "togglegroup"},
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
            grid-template-columns: minmax(220px, 1fr) repeat(7, minmax(120px, 170px));
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
        .comparison-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
        }
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
            .comparison-grid { grid-template-columns: 1fr; }
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
                            html.Label("Famille A"),
                            dcc.Dropdown(id="group-dropdown", clearable=False),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Trace A"),
                            dcc.Dropdown(id="trace-dropdown", clearable=False),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Famille B"),
                            dcc.Dropdown(id="compare-group-dropdown", clearable=False),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Trace B"),
                            dcc.Dropdown(id="compare-trace-dropdown", clearable=False),
                        ]
                    ),
                ],
                className="topbar",
            ),
            dcc.Interval(
                id="refresh-interval",
                interval=5000,
                n_intervals=0,
                disabled=True,
            ),
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
                                                        "Signal A",
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
                                                        "Signal B",
                                                        className="panel-title",
                                                    )
                                                ],
                                                className="panel-header",
                                            ),
                                            dcc.Graph(
                                                id="compare-time-graph",
                                                config={"displayModeBar": True},
                                                style={"height": "360px"},
                                            ),
                                        ],
                                        className="panel",
                                    ),
                                ],
                                className="comparison-grid",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H2(
                                                        "Spectrogramme A",
                                                        className="panel-title",
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label("STFT"),
                                                            dcc.Dropdown(
                                                                id="spectrogram-stft-mode-dropdown",
                                                                options=[
                                                                    {"label": "Benchmark", "value": "benchmark"},
                                                                    {"label": "Manuelle", "value": "manual"},
                                                                ],
                                                                value="benchmark",
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        style={"width": "130px"},
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label("Fenetre"),
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
                                            html.Div(
                                                id="spectrogram-stft-caption",
                                                className="muted panel-body",
                                            ),
                                        ],
                                        className="panel",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H2(
                                                        "Spectrogramme B",
                                                        className="panel-title",
                                                    )
                                                ],
                                                className="panel-header",
                                            ),
                                            dcc.Graph(
                                                id="compare-spectrogram-graph",
                                                config={"displayModeBar": True},
                                                style={"height": "390px"},
                                            ),
                                        ],
                                        className="panel",
                                    ),
                                ],
                                className="comparison-grid",
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
                                                                {"label": "Energie masquee", "value": "masked_energy"},
                                                                {"label": "Bins actifs EM", "value": "active"},
                                                                {"label": "Correct vs GT", "value": "gt_error"},
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
                                            html.Div(
                                                [
                                                    html.Label("GT"),
                                                    dcc.Dropdown(
                                                        id="sawada-gt-permutation-dropdown",
                                                        options=[
                                                            {"label": "Normale", "value": "identity"},
                                                            {"label": "Permutee 1/2", "value": "swap"},
                                                        ],
                                                        value="identity",
                                                        clearable=False,
                                                    ),
                                                ],
                                                style={"width": "150px"},
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
                                        [
                                            html.H2("Plans complexes EM", className="panel-title"),
                                            html.Div(
                                                [
                                                    html.Label("Frequence"),
                                                    dcc.Dropdown(
                                                        id="complex-frequency-dropdown",
                                                        value=0,
                                                        clearable=False,
                                                    ),
                                                ],
                                                style={"width": "190px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("Vecteurs"),
                                                    dcc.Dropdown(
                                                        id="complex-vector-space-dropdown",
                                                        options=[
                                                            {"label": "Blanchis", "value": "whitened"},
                                                            {"label": "Non blanchis", "value": "unwhitened"},
                                                        ],
                                                        value="whitened",
                                                        clearable=False,
                                                    ),
                                                ],
                                                style={"width": "150px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("Coord."),
                                                    dcc.Dropdown(
                                                        id="complex-coordinate-mode-dropdown",
                                                        options=[
                                                            {"label": "M*conj(M1)", "value": "relative"},
                                                            {"label": "Directes", "value": "direct"},
                                                        ],
                                                        value="relative",
                                                        clearable=False,
                                                    ),
                                                ],
                                                style={"width": "160px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("Couleur"),
                                                    dcc.Dropdown(
                                                        id="complex-color-mode-dropdown",
                                                        options=[
                                                            {"label": "Source estimee", "value": "source"},
                                                            {"label": "Correct vs GT", "value": "gt"},
                                                        ],
                                                        value="source",
                                                        clearable=False,
                                                    ),
                                                ],
                                                style={"width": "160px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("Hors EM"),
                                                    dcc.Dropdown(
                                                        id="complex-rejected-bins-dropdown",
                                                        options=[
                                                            {"label": "Masquer", "value": "hide"},
                                                            {"label": "Afficher", "value": "show"},
                                                        ],
                                                        value="hide",
                                                        clearable=False,
                                                    ),
                                                ],
                                                style={"width": "130px"},
                                            ),
                                        ],
                                        className="panel-header",
                                    ),
                                    dcc.Graph(
                                        id="sawada-complex-plane-graph",
                                        config={"displayModeBar": True},
                                        style={"height": "520px"},
                                    ),
                                    html.Div(
                                        id="sawada-complex-plane-caption",
                                        className="muted panel-body",
                                    ),
                                ],
                                className="panel",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H2("Phases centroides", className="panel-title"),
                                            html.Div(
                                                [
                                                    html.Label("Couleur"),
                                                    dcc.Dropdown(
                                                        id="centroid-phase-color-dropdown",
                                                        options=[
                                                            {"label": "Frequence", "value": "frequency"},
                                                            {"label": "Source attribuee", "value": "source"},
                                                            {"label": "Frequence + source", "value": "source_frequency"},
                                                        ],
                                                        value="frequency",
                                                        clearable=False,
                                                    ),
                                                ],
                                                style={"width": "180px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("Vecteurs"),
                                                    dcc.Dropdown(
                                                        id="centroid-phase-reference-dropdown",
                                                        options=[
                                                            {"label": "Cm*conj(M1)", "value": "relative"},
                                                            {"label": "Cm", "value": "direct"},
                                                        ],
                                                        value="relative",
                                                        clearable=False,
                                                    ),
                                                ],
                                                style={"width": "140px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("Angle"),
                                                    dcc.Dropdown(
                                                        id="centroid-phase-angle-dropdown",
                                                        options=[
                                                            {"label": "Delta precedent", "value": "delta_previous"},
                                                            {"label": "arg", "value": "raw"},
                                                        ],
                                                        value="delta_previous",
                                                        clearable=False,
                                                    ),
                                                ],
                                                style={"width": "130px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("f min"),
                                                    dcc.Input(
                                                        id="centroid-phase-frequency-min-input",
                                                        type="number",
                                                        placeholder="Hz",
                                                        debounce=True,
                                                        style={"width": "100%"},
                                                    ),
                                                ],
                                                style={"width": "100px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("f max"),
                                                    dcc.Input(
                                                        id="centroid-phase-frequency-max-input",
                                                        type="number",
                                                        placeholder="Hz",
                                                        debounce=True,
                                                        style={"width": "100%"},
                                                    ),
                                                ],
                                                style={"width": "100px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("GT centroide"),
                                                    dcc.Dropdown(
                                                        id="centroid-gt-mode-dropdown",
                                                        options=[
                                                            {
                                                                "label": "Melange masque",
                                                                "value": "mixture_masked",
                                                            },
                                                            {
                                                                "label": "Source directe",
                                                                "value": "source_direct",
                                                            },
                                                        ],
                                                        value="mixture_masked",
                                                        clearable=False,
                                                    ),
                                                ],
                                                style={"width": "170px"},
                                            ),
                                        ],
                                        className="panel-header",
                                    ),
                                    dcc.Graph(
                                        id="sawada-centroid-phase-graph",
                                        config={"displayModeBar": True},
                                        style={"height": "520px"},
                                    ),
                                    html.Div(
                                        id="sawada-centroid-phase-caption",
                                        className="muted panel-body",
                                    ),
                                    dcc.Graph(
                                        id="sawada-centroid-argument-frequency-graph",
                                        config={"displayModeBar": True},
                                        style={"height": "520px"},
                                    ),
                                    html.Div(
                                        id="sawada-centroid-argument-frequency-caption",
                                        className="muted panel-body",
                                    ),
                                ],
                                className="panel",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H2("RANSAC circulaire", className="panel-title"),
                                            html.Div(
                                                [
                                                    html.Label("Source"),
                                                    dcc.Dropdown(
                                                        id="ransac-source-dropdown",
                                                        value=0,
                                                        clearable=False,
                                                    ),
                                                ],
                                                style={"width": "140px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("Couleur"),
                                                    dcc.Dropdown(
                                                        id="ransac-color-dropdown",
                                                        options=[
                                                            {"label": "Attribution", "value": "assignment"},
                                                            {"label": "Distance", "value": "distance"},
                                                        ],
                                                        value="assignment",
                                                        clearable=False,
                                                    ),
                                                ],
                                                style={"width": "150px"},
                                            ),
                                        ],
                                        className="panel-header",
                                    ),
                                    dcc.Graph(
                                        id="sawada-ransac-graph",
                                        config={"displayModeBar": True},
                                        style={"height": "720px"},
                                    ),
                                    html.Div(
                                        id="sawada-ransac-caption",
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
        Output("compare-group-dropdown", "options"),
        Output("compare-group-dropdown", "value"),
        Input("split-dropdown", "value"),
        Input("scene-dropdown", "value"),
        Input("algorithm-dropdown", "value"),
        State("compare-group-dropdown", "value"),
    )
    def update_compare_groups(
        split: str, scene_id: str, algorithm: str, current_group: str | None
    ) -> tuple[list[dict[str, str]], str]:
        if not split or not scene_id or not algorithm:
            return [], ""
        groups = _signal_groups(_bundle(config, split, scene_id, algorithm))
        keys = list(groups)
        if current_group in groups:
            value = current_group
        elif "estimated" in groups:
            value = "estimated"
        elif len(keys) > 1:
            value = keys[1]
        else:
            value = keys[0] if keys else ""
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
        Output("compare-trace-dropdown", "options"),
        Output("compare-trace-dropdown", "value"),
        Input("split-dropdown", "value"),
        Input("scene-dropdown", "value"),
        Input("algorithm-dropdown", "value"),
        Input("compare-group-dropdown", "value"),
        State("compare-trace-dropdown", "value"),
    )
    def update_compare_traces(
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
        Output("complex-frequency-dropdown", "options"),
        Output("complex-frequency-dropdown", "value"),
        Input("split-dropdown", "value"),
        Input("scene-dropdown", "value"),
        Input("algorithm-dropdown", "value"),
        State("complex-frequency-dropdown", "value"),
    )
    def update_complex_frequencies(
        split: str,
        scene_id: str,
        algorithm: str,
        current_frequency: int | None,
    ) -> tuple[list[dict[str, int]], int]:
        if not split or not scene_id or algorithm != "sawada":
            return [], 0

        model = _bundle(config, split, scene_id, algorithm)["sawada_model"]
        bin_vectors = np.asarray(model.get("bin_vectors", []))
        frequencies = np.asarray(model.get("frequencies", []), dtype=float)
        if bin_vectors.ndim != 3 or bin_vectors.size == 0:
            return [], 0

        n_freqs = bin_vectors.shape[1]
        frequency_indexes = list(range(n_freqs))
        if current_frequency in frequency_indexes:
            value = int(current_frequency)
        else:
            frequency_energy = np.asarray(model.get("frequency_energy", []), dtype=float)
            value = (
                int(np.nanargmax(frequency_energy))
                if frequency_energy.ndim == 1 and frequency_energy.size == n_freqs
                else 0
            )

        options = []
        for frequency_index in frequency_indexes:
            if frequencies.size > frequency_index:
                label = f"{frequency_index} - {float(frequencies[frequency_index]):.1f} Hz"
            else:
                label = str(frequency_index)
            options.append({"label": label, "value": frequency_index})
        return options, value

    @app.callback(
        Output("ransac-source-dropdown", "options"),
        Output("ransac-source-dropdown", "value"),
        Input("split-dropdown", "value"),
        Input("scene-dropdown", "value"),
        Input("algorithm-dropdown", "value"),
        State("ransac-source-dropdown", "value"),
    )
    def update_ransac_controls(
        split: str,
        scene_id: str,
        algorithm: str,
        current_source: int | None,
    ) -> tuple[list[dict[str, int]], int]:
        if not split or not scene_id or algorithm != "sawada":
            return [], 0

        model = _bundle(config, split, scene_id, algorithm)["sawada_model"]
        slopes = np.asarray(model.get("source_assignment_slopes", []), dtype=float)
        labels = np.asarray(model.get("source_assignment_labels", []), dtype=int)
        masks = np.asarray(model.get("masks", []))

        if slopes.ndim == 2 and slopes.size:
            n_sources = slopes.shape[0]
        elif labels.ndim == 2 and labels.size:
            valid_labels = labels[labels >= 0]
            n_sources = int(np.max(valid_labels)) + 1 if valid_labels.size else 0
        elif masks.ndim == 3 and masks.size:
            n_sources = masks.shape[0]
        else:
            n_sources = 0

        source_indexes = list(range(n_sources))
        source_value = current_source if current_source in source_indexes else 0
        return (
            [
                {"label": f"Source {source_index + 1}", "value": source_index}
                for source_index in source_indexes
            ],
            int(source_value),
        )

    @app.callback(
        Output("sawada-mask-graph", "figure"),
        Output("sawada-mask-caption", "children"),
        Input("split-dropdown", "value"),
        Input("scene-dropdown", "value"),
        Input("algorithm-dropdown", "value"),
        Input("mask-source-dropdown", "value"),
        Input("mask-frequency-scale-dropdown", "value"),
        Input("sawada-map-kind-dropdown", "value"),
        Input("sawada-gt-permutation-dropdown", "value"),
    )
    def update_sawada_mask(
        split: str,
        scene_id: str,
        algorithm: str,
        source_index: int,
        frequency_scale: str,
        map_kind: str,
        gt_permutation: str,
    ) -> tuple[go.Figure, str]:
        if not split or not scene_id or algorithm != "sawada":
            return _sawada_mask_figure({}, 0), "Disponible uniquement pour Sawada."

        bundle = _bundle(config, split, scene_id, algorithm)
        model = bundle["sawada_model"]
        masks = np.asarray(model.get("masks", []))
        if masks.ndim != 3 or masks.size == 0:
            return (
                _sawada_mask_figure({}, 0),
                "Aucun sawada_model.npz trouve pour cette scene. Relance le benchmark Sawada.",
            )

        source_index = min(max(int(source_index or 0), 0), masks.shape[0] - 1)
        if map_kind == "active":
            active_tf_mask = np.asarray(model.get("active_tf_mask", []), dtype=float)
            if active_tf_mask.ndim != 2 or active_tf_mask.size == 0:
                return (
                    _sawada_mask_figure(model, source_index, frequency_scale, map_kind),
                    "Masque des bins actifs absent. Relance le benchmark Sawada avec cette version.",
                )
            threshold = np.asarray(model.get("energy_threshold_db", np.nan)).item()
            threshold_text = "-" if np.isnan(threshold) else f"{float(threshold):.1f} dB"
            caption = (
                f"Bins actifs EM: {float(np.mean(active_tf_mask)):.1%} des bins gardes, "
                f"seuil {threshold_text}."
            )
        elif map_kind == "masked_energy":
            tf_energy = np.asarray(model.get("tf_energy", []), dtype=float)
            if tf_energy.ndim != 2 or tf_energy.size == 0:
                return (
                    _sawada_mask_figure(model, source_index, frequency_scale, map_kind),
                    "Energie TF absente. Relance le benchmark Sawada avec cette version.",
                )
            n_freqs = min(masks.shape[1], tf_energy.shape[0])
            n_times = min(masks.shape[2], tf_energy.shape[1])
            masked_energy = masks[source_index, :n_freqs, :n_times] * tf_energy[:n_freqs, :n_times]
            total_energy = float(np.sum(tf_energy[:n_freqs, :n_times]))
            source_energy = float(np.sum(masked_energy))
            ratio_text = "-" if total_energy <= 0 else f"{source_energy / total_energy:.1%}"
            caption = (
                f"Energie masquee source {source_index + 1}: "
                f"{n_freqs} bins frequences x {n_times} trames, "
                f"{ratio_text} de l'energie TF totale du melange."
            )
        elif map_kind == "gt_error":
            gt_correctness = _sawada_gt_correctness(bundle, model, gt_permutation)
            if gt_correctness.ndim != 2 or gt_correctness.size == 0:
                return (
                    _sawada_mask_figure(model, source_index, frequency_scale, map_kind),
                    "GT indisponible. Verifie que le dataset original est bien charge.",
                )
            valid = gt_correctness != 0
            correct_ratio = (
                float(np.mean(gt_correctness[valid] > 0))
                if np.any(valid)
                else float("nan")
            )
            ratio_text = "-" if np.isnan(correct_ratio) else f"{correct_ratio:.1%}"
            caption = (
                f"Comparaison au spectro source dominant: {int(np.sum(valid))} bins compares, "
                f"{ratio_text} corrects. Permutation GT: "
                f"{'1/2 inversee' if gt_permutation == 'swap' else 'normale'}."
            )
            return (
                _sawada_mask_figure(
                    model,
                    source_index,
                    frequency_scale,
                    map_kind,
                    gt_correctness=gt_correctness,
                ),
                caption,
            )
        elif map_kind == "posterior":
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

        threshold = np.asarray(model.get("energy_threshold_db", np.nan)).item()
        active_tf_mask = np.asarray(model.get("active_tf_mask", []), dtype=float)
        active_frequency_mask = np.asarray(model.get("active_frequency_mask", []), dtype=bool)
        frequency_energy = np.mean(energy, axis=1)
        db = 10 * np.log10(frequency_energy + 1e-20)
        quiet_threshold = np.nanpercentile(db, 10)
        quiet_count = int(np.sum(db <= quiet_threshold))
        threshold_text = "-" if np.isnan(threshold) else f"{float(threshold):.1f} dB"
        active_text = (
            "-"
            if active_tf_mask.ndim != 2 or active_tf_mask.size == 0
            else f"{float(np.mean(active_tf_mask)):.1%}"
        )
        active_frequency_text = (
            "-"
            if active_frequency_mask.ndim != 1 or active_frequency_mask.size == 0
            else f"{int(np.sum(active_frequency_mask))}/{active_frequency_mask.size}"
        )
        caption = (
            f"Energie brute avant normalisation: {energy.shape[0]} bins frequences x "
            f"{energy.shape[1]} trames. Seuil EM {threshold_text}, "
            f"bins actifs apres filtrage temporel {active_text}, "
            f"frequences utilisees par l'EM {active_frequency_text}. "
            f"{quiet_count} bins frequences sont dans les 10% les plus faibles."
        )
        return _sawada_energy_figure(model, frequency_scale), caption

    @app.callback(
        Output("sawada-complex-plane-graph", "figure"),
        Output("sawada-complex-plane-caption", "children"),
        Input("split-dropdown", "value"),
        Input("scene-dropdown", "value"),
        Input("algorithm-dropdown", "value"),
        Input("complex-frequency-dropdown", "value"),
        Input("complex-vector-space-dropdown", "value"),
        Input("complex-coordinate-mode-dropdown", "value"),
        Input("complex-color-mode-dropdown", "value"),
        Input("complex-rejected-bins-dropdown", "value"),
        Input("sawada-gt-permutation-dropdown", "value"),
    )
    def update_sawada_complex_plane(
        split: str,
        scene_id: str,
        algorithm: str,
        frequency_index: int,
        vector_space: str,
        coordinate_mode: str,
        color_mode: str,
        rejected_bins_mode: str,
        gt_permutation: str,
    ) -> tuple[go.Figure, str]:
        if not split or not scene_id or algorithm != "sawada":
            return _sawada_complex_plane_figure({}, 0), "Disponible uniquement pour Sawada."

        vector_space = vector_space or "whitened"
        coordinate_mode = coordinate_mode or "relative"
        color_mode = color_mode or "source"
        show_rejected_bins = (rejected_bins_mode or "hide") == "show"
        bundle = _bundle(config, split, scene_id, algorithm)
        model = bundle["sawada_model"]
        vector_key = "bin_vectors_unwhitened" if vector_space == "unwhitened" else "bin_vectors"
        centroid_key = "centroids_unwhitened" if vector_space == "unwhitened" else "centroids"
        bin_vectors = np.asarray(model.get(vector_key, []))
        masks = np.asarray(model.get("masks", []), dtype=float)
        active_tf_mask = np.asarray(model.get("active_tf_mask", []), dtype=bool)
        active_frequency_mask = np.asarray(model.get("active_frequency_mask", []), dtype=bool)
        active_clusters = np.asarray(model.get("active_clusters", []), dtype=bool)
        centroids = np.asarray(model.get(centroid_key, []))
        if bin_vectors.ndim != 3 or bin_vectors.size == 0:
            return (
                _sawada_complex_plane_figure(
                    model,
                    frequency_index,
                    vector_space,
                    coordinate_mode,
                    color_mode,
                    show_rejected_bins,
                ),
                "Vecteurs complexes absents. Relance le benchmark Sawada avec cette version.",
            )
        if masks.ndim != 3 or masks.size == 0 or centroids.ndim != 3 or centroids.size == 0:
            return (
                _sawada_complex_plane_figure(
                    model,
                    frequency_index,
                    vector_space,
                    coordinate_mode,
                    color_mode,
                    show_rejected_bins,
                ),
                "Masques ou centroides absents dans sawada_model.npz.",
            )

        n_mics, n_freqs, n_times = bin_vectors.shape
        frequency_index = min(max(int(frequency_index or 0), 0), n_freqs - 1)
        frequencies = np.asarray(model.get("frequencies", []), dtype=float)
        frequency_text = (
            f"{float(frequencies[frequency_index]):.1f} Hz"
            if frequencies.size > frequency_index
            else f"bin {frequency_index}"
        )
        source_masks = masks[:, frequency_index, :]
        used_by_em = (
            active_tf_mask[frequency_index]
            if active_tf_mask.ndim == 2 and active_tf_mask.shape[0] > frequency_index
            else np.max(source_masks, axis=0) > 0.5
        )
        assigned = np.argmax(source_masks, axis=0)
        assigned_strength = np.max(source_masks, axis=0)
        has_assignment = assigned_strength > 0.5
        active_cluster_count = (
            int(np.sum(active_clusters[frequency_index]))
            if active_clusters.ndim == 2 and active_clusters.shape[0] > frequency_index
            else source_masks.shape[0]
        )
        frequency_used_text = (
            "oui"
            if active_frequency_mask.ndim == 1
            and active_frequency_mask.size > frequency_index
            and active_frequency_mask[frequency_index]
            else "non"
            if active_frequency_mask.ndim == 1
            and active_frequency_mask.size > frequency_index
            else "inconnu"
        )
        gt_correctness = (
            _sawada_gt_correctness(bundle, model, gt_permutation)
            if color_mode == "gt"
            else np.empty((0, 0), dtype=float)
        )
        counts = [
            f"S{source_index + 1}: "
            f"{int(np.sum(used_by_em & has_assignment & (assigned == source_index)))}"
            for source_index in range(source_masks.shape[0])
        ]
        inactive_count = int(np.sum(~has_assignment))
        rejected_count = int(np.sum(~used_by_em))
        point_counts = counts + [f"Hors EM: {rejected_count}"]
        gt_text = ""
        if color_mode == "gt":
            if gt_correctness.ndim == 2 and gt_correctness.shape[0] > frequency_index:
                correctness = gt_correctness[frequency_index, :n_times]
                valid = correctness != 0
                correct_count = int(np.sum(correctness > 0))
                wrong_count = int(np.sum(correctness < 0))
                gt_text = (
                    f" GT: {correct_count} corrects, {wrong_count} faux, "
                    f"{int(np.sum(~valid))} ignores."
                )
            else:
                gt_text = " GT indisponible."
        caption = (
            f"Ligne frequencielle {frequency_index} ({frequency_text}): {n_times} trames x "
            f"{n_mics} composantes complexes. Points: {', '.join(point_counts)}. "
            f"Frequence utilisee par l'EM: {frequency_used_text}. "
            f"Clusters actifs: {active_cluster_count}/{source_masks.shape[0]}. "
            f"Non utilises par l'EM: {inactive_count}. "
            f"Hors seuil energie: {rejected_count}"
            f"{' affiches' if show_rejected_bins else ' masques'}. "
            f"Repere: {'non blanchi' if vector_space == 'unwhitened' else 'blanchi'}, "
            f"{'M*conj(M1)' if coordinate_mode == 'relative' else 'composantes directes'}."
            f"{gt_text}"
        )
        return (
            _sawada_complex_plane_figure(
                model,
                frequency_index,
                vector_space,
                coordinate_mode,
                color_mode,
                show_rejected_bins,
                gt_correctness=gt_correctness,
            ),
            caption,
        )

    @app.callback(
        Output("sawada-centroid-phase-graph", "figure"),
        Output("sawada-centroid-phase-caption", "children"),
        Input("split-dropdown", "value"),
        Input("scene-dropdown", "value"),
        Input("algorithm-dropdown", "value"),
        Input("centroid-phase-color-dropdown", "value"),
        Input("centroid-phase-reference-dropdown", "value"),
        Input("centroid-phase-angle-dropdown", "value"),
        Input("centroid-phase-frequency-min-input", "value"),
        Input("centroid-phase-frequency-max-input", "value"),
    )
    def update_sawada_centroid_phases(
        split: str,
        scene_id: str,
        algorithm: str,
        color_mode: str,
        reference_mode: str,
        angle_mode: str,
        frequency_min: float | None,
        frequency_max: float | None,
    ) -> tuple[go.Figure, str]:
        if not split or not scene_id or algorithm != "sawada":
            return (
                _sawada_centroid_phase_figure(
                    {},
                    color_mode,
                    reference_mode,
                    angle_mode,
                    frequency_min,
                    frequency_max,
                ),
                "Disponible uniquement pour Sawada.",
            )

        bundle = _bundle(config, split, scene_id, algorithm)
        model = bundle["sawada_model"]
        centroids = np.asarray(model.get("centroids_unwhitened", []))
        centroid_space = "non blanchi"
        if centroids.ndim != 3 or centroids.size == 0:
            centroids = np.asarray(model.get("centroids", []))
            centroid_space = "blanchi"
        if centroids.ndim != 3 or centroids.size == 0:
            return (
                _sawada_centroid_phase_figure(
                    model,
                    color_mode,
                    reference_mode,
                    angle_mode,
                    frequency_min,
                    frequency_max,
                ),
                "Centroides absents dans sawada_model.npz.",
            )

        frequencies = np.asarray(model.get("frequencies", []), dtype=float)
        if frequencies.size == centroids.shape[1] and frequencies.size != centroids.shape[0]:
            centroids = np.moveaxis(centroids, 1, 0)

        n_freqs, n_mics, n_sources = centroids.shape
        assignment_labels = np.asarray(
            model.get("source_assignment_labels", []),
            dtype=int,
        )
        use_final_assignment = assignment_labels.shape == (n_freqs, n_sources)
        if frequencies.size < n_freqs:
            caption_frequencies = np.arange(n_freqs, dtype=float)
        else:
            caption_frequencies = frequencies[:n_freqs]
        use_delta_previous = (angle_mode or "delta_previous") == "delta_previous"
        use_relative_phase = (reference_mode or "relative") == "relative"
        frequency_min = None if frequency_min is None else float(frequency_min)
        frequency_max = None if frequency_max is None else float(frequency_max)
        if (
            frequency_min is not None
            and frequency_max is not None
            and frequency_min > frequency_max
        ):
            frequency_min, frequency_max = frequency_max, frequency_min
        frequency_band = np.isfinite(caption_frequencies)
        if frequency_min is not None:
            frequency_band &= caption_frequencies >= frequency_min
        if frequency_max is not None:
            frequency_band &= caption_frequencies <= frequency_max
        if use_delta_previous:
            frequency_band &= np.arange(n_freqs) > 0
        valid_reference = np.ones((n_freqs, n_sources), dtype=bool)
        point_count = int(np.sum(frequency_band[:, np.newaxis] & valid_reference))
        if (color_mode or "frequency") == "frequency":
            color_text = "frequence"
        elif color_mode == "source_frequency":
            color_text = "frequence avec une palette differente par source"
        else:
            color_text = "source attribuee"
        vector_text = "Cm*conj(M1)" if use_relative_phase else "Cm"
        theta_text = (
            f"arg({vector_text}(f_i)) - arg({vector_text}(f_i-1))"
            if use_delta_previous
            else f"arg({vector_text})"
        )
        if frequency_min is None and frequency_max is None:
            band_text = "toutes frequences"
        elif frequency_min is None:
            band_text = f"f <= {frequency_max:g} Hz"
        elif frequency_max is None:
            band_text = f"f >= {frequency_min:g} Hz"
        else:
            band_text = f"{frequency_min:g} <= f <= {frequency_max:g} Hz"
        caption = (
            f"{n_freqs} lignes frequentielles x {n_sources} sources x {n_mics} composantes. "
            f"{point_count} centroides traces par composante"
            f"{' apres exclusion du premier bin frequentiel' if use_delta_previous else ''}. "
            f"Bande: {band_text}. "
            f"Repere: {centroid_space}, theta = {theta_text}. "
            "Chaque point est projete sur le cercle unite avec (cos(theta), sin(theta)). "
            f"Couleur: {color_text}."
        )
        return _sawada_centroid_phase_figure(
            model,
            color_mode,
            reference_mode,
            angle_mode,
            frequency_min,
            frequency_max,
        ), caption

    @app.callback(
        Output("sawada-centroid-argument-frequency-graph", "figure"),
        Output("sawada-centroid-argument-frequency-caption", "children"),
        Input("split-dropdown", "value"),
        Input("scene-dropdown", "value"),
        Input("algorithm-dropdown", "value"),
        Input("centroid-phase-frequency-min-input", "value"),
        Input("centroid-phase-frequency-max-input", "value"),
        Input("sawada-gt-permutation-dropdown", "value"),
        Input("centroid-gt-mode-dropdown", "value"),
    )
    def update_sawada_centroid_argument_frequency(
        split: str,
        scene_id: str,
        algorithm: str,
        frequency_min: float | None,
        frequency_max: float | None,
        gt_permutation: str,
        gt_centroid_mode: str,
    ) -> tuple[go.Figure, str]:
        if not split or not scene_id or algorithm != "sawada":
            return (
                _sawada_centroid_argument_frequency_figure({}, frequency_min, frequency_max),
                "Disponible uniquement pour Sawada.",
            )

        bundle = _bundle(config, split, scene_id, algorithm)
        model = bundle["sawada_model"]
        centroids = np.asarray(model.get("centroids_unwhitened", []))
        centroid_space = "non blanchi"
        if centroids.ndim != 3 or centroids.size == 0:
            centroids = np.asarray(model.get("centroids", []))
            centroid_space = "blanchi"
        relative_phases = np.asarray(
            model.get("source_assignment_relative_phases", []),
            dtype=float,
        )
        has_complex_centroids = centroids.ndim == 3 and centroids.size > 0
        has_relative_phases = relative_phases.ndim == 3 and relative_phases.size > 0
        if not has_complex_centroids and not has_relative_phases:
            return (
                _sawada_centroid_argument_frequency_figure(model, frequency_min, frequency_max),
                "Centroides et phases RANSAC absents dans sawada_model.npz.",
            )

        frequencies = np.asarray(model.get("frequencies", []), dtype=float)
        if has_complex_centroids:
            if frequencies.size == centroids.shape[1] and frequencies.size != centroids.shape[0]:
                centroids = np.moveaxis(centroids, 1, 0)
            n_freqs, n_mics, n_sources = centroids.shape
        else:
            n_freqs, n_sources, n_mics = relative_phases.shape
            centroid_space = "phases RANSAC sauvegardees"
        assignment_labels = np.asarray(
            model.get("source_assignment_labels", []),
            dtype=int,
        )
        use_final_assignment = assignment_labels.shape == (n_freqs, n_sources)
        if frequencies.size < n_freqs:
            caption_frequencies = np.arange(n_freqs, dtype=float)
        else:
            caption_frequencies = frequencies[:n_freqs]
        frequency_min = None if frequency_min is None else float(frequency_min)
        frequency_max = None if frequency_max is None else float(frequency_max)
        if (
            frequency_min is not None
            and frequency_max is not None
            and frequency_min > frequency_max
        ):
            frequency_min, frequency_max = frequency_max, frequency_min
        frequency_band = np.isfinite(caption_frequencies)
        if frequency_min is not None:
            frequency_band &= caption_frequencies >= frequency_min
        if frequency_max is not None:
            frequency_band &= caption_frequencies <= frequency_max
        point_count = int(np.sum(frequency_band) * n_sources)
        gt_centroids = _ground_truth_centroids(
            bundle,
            model,
            gt_permutation,
            gt_centroid_mode,
        )
        gt_available = gt_centroids.ndim == 3 and gt_centroids.size > 0
        gt_point_count = 0
        if gt_available:
            gt_n_freqs = min(gt_centroids.shape[0], caption_frequencies.size)
            gt_point_count = int(np.sum(frequency_band[:gt_n_freqs]) * gt_centroids.shape[2])
        gt_mode_text = (
            "source directe"
            if gt_centroid_mode == "source_direct"
            else "melange masque"
        )
        if frequency_min is None and frequency_max is None:
            band_text = "toutes frequences"
        elif frequency_min is None:
            band_text = f"f <= {frequency_max:g} Hz"
        elif frequency_max is None:
            band_text = f"f >= {frequency_min:g} Hz"
        else:
            band_text = f"{frequency_min:g} <= f <= {frequency_max:g} Hz"
        caption = (
            f"{n_freqs} lignes frequentielles x {n_sources} sources x {n_mics} composantes. "
            f"{point_count} centroides traces par composante. "
            f"Bande: {band_text}. Repere: {centroid_space}, "
            "arg(Cm*conj(M1)) en fonction de la frequence. "
            f"Couleur Sawada: {'source finale RANSAC' if use_final_assignment else 'cluster EM local'}. "
            f"GT: {'disponible' if gt_available else 'indisponible'}"
            f"{f', {gt_point_count} centroides GT traces par composante' if gt_available else ''}. "
            f"Mode GT: {gt_mode_text}. "
            f"Permutation GT: {'1/2 inversee' if gt_permutation == 'swap' else 'normale'}."
        )
        return (
            _sawada_centroid_argument_frequency_figure(
                model,
                frequency_min,
                frequency_max,
                gt_centroids=gt_centroids,
            ),
            caption,
        )

    @app.callback(
        Output("sawada-ransac-graph", "figure"),
        Output("sawada-ransac-caption", "children"),
        Input("split-dropdown", "value"),
        Input("scene-dropdown", "value"),
        Input("algorithm-dropdown", "value"),
        Input("ransac-source-dropdown", "value"),
        Input("ransac-color-dropdown", "value"),
    )
    def update_sawada_ransac(
        split: str,
        scene_id: str,
        algorithm: str,
        source_index: int,
        color_mode: str,
    ) -> tuple[go.Figure, str]:
        if not split or not scene_id or algorithm != "sawada":
            return _sawada_ransac_figure({}, source_index), "Disponible uniquement pour Sawada."

        model = _bundle(config, split, scene_id, algorithm)["sawada_model"]
        relative_phases = np.asarray(
            model.get("source_assignment_relative_phases", []),
            dtype=float,
        )
        labels = np.asarray(model.get("source_assignment_labels", []), dtype=int)
        selected_labels = np.asarray(
            model.get("source_assignment_selected_labels", []),
            dtype=int,
        )
        distances = np.asarray(model.get("source_assignment_distances", []), dtype=float)
        frequencies = np.asarray(model.get("frequencies", []), dtype=float)
        slopes = np.asarray(model.get("source_assignment_slopes", []), dtype=float)
        scores = np.asarray(model.get("source_assignment_scores", []), dtype=float)
        n_inliers = np.asarray(model.get("source_assignment_n_inliers", []), dtype=float)
        n_trials = np.asarray(model.get("source_assignment_n_trials", []), dtype=float)
        converged = np.asarray(model.get("source_assignment_converged", []), dtype=bool)
        frequency_inliers = np.asarray(
            model.get("source_assignment_frequency_inliers", []),
            dtype=bool,
        )
        selected_centroids = np.asarray(
            model.get("source_assignment_selected_centroids", []),
            dtype=int,
        )

        if (
            relative_phases.ndim != 3
            or labels.shape != relative_phases.shape[:2]
            or slopes.ndim != 2
            or slopes.size == 0
        ):
            return (
                _sawada_ransac_figure(model, source_index, color_mode),
                "RANSAC absent dans sawada_model.npz. Relance le benchmark Sawada avec cette version.",
            )

        n_freqs, n_centroids, n_components = relative_phases.shape
        n_sources = slopes.shape[0]
        if frequencies.size < n_freqs:
            frequencies = np.arange(n_freqs, dtype=float)
        else:
            frequencies = frequencies[:n_freqs]
        source_index = min(max(int(source_index or 0), 0), n_sources - 1)

        available_points = labels >= 0
        if selected_labels.shape == labels.shape:
            available_points |= selected_labels >= 0
        if distances.ndim == 3 and distances.shape[1:] == labels.shape:
            available_points |= np.any(np.isfinite(distances), axis=0)
        elif distances.shape == labels.shape:
            available_points |= np.isfinite(distances)
        active_frequencies = np.any(available_points, axis=1)
        active_frequency_count = int(np.sum(active_frequencies))
        max_active_frequency = (
            float(np.nanmax(frequencies[active_frequencies & np.isfinite(frequencies)]))
            if np.any(active_frequencies & np.isfinite(frequencies))
            else float("nan")
        )
        max_frequency_text = (
            "-"
            if np.isnan(max_active_frequency)
            else f"{max_active_frequency:.1f} Hz"
        )
        counts = [
            f"S{label_index + 1}: {int(np.sum(labels == label_index))}"
            for label_index in range(n_sources)
        ]
        rejected_count = int(np.sum(labels < 0))
        finite_count = int(
            np.sum(
                np.isfinite(relative_phases)
                & available_points[:, :, np.newaxis]
            )
        )

        selected_inlier_count = 0
        if (
            frequency_inliers.ndim == 2
            and selected_centroids.ndim == 2
            and source_index < frequency_inliers.shape[0]
            and source_index < selected_centroids.shape[0]
        ):
            n_selected_freqs = min(n_freqs, frequency_inliers.shape[1], selected_centroids.shape[1])
            selected = selected_centroids[source_index, :n_selected_freqs]
            selected_inlier_count = int(
                np.sum(
                    frequency_inliers[source_index, :n_selected_freqs]
                    & (selected >= 0)
                    & (selected < n_centroids)
                )
            )

        score_text = (
            "-"
            if scores.ndim != 1 or scores.size <= source_index or not np.isfinite(scores[source_index])
            else f"{float(scores[source_index]):.4g}"
        )
        inliers_text = (
            "-"
            if n_inliers.ndim != 1 or n_inliers.size <= source_index
            else str(int(n_inliers[source_index]))
        )
        trials_text = (
            "-"
            if n_trials.ndim != 1 or n_trials.size <= source_index
            else str(int(n_trials[source_index]))
        )
        converged_text = (
            "-"
            if converged.ndim != 1 or converged.size <= source_index
            else ("oui" if bool(converged[source_index]) else "non")
        )
        caption = (
            f"RANSAC sur {n_freqs} frequences x {n_centroids} centroides x "
            f"{n_components} composantes de phase relative. "
            f"Frequences avec centroides disponibles: {active_frequency_count}, "
            f"max {max_frequency_text}. "
            "Toutes les composantes sont affichees simultanement. "
            f"Attributions finales: {', '.join(counts)}, non attribues: {rejected_count}. "
            f"Source affichee S{source_index + 1}: score {score_text}, "
            f"inliers declares {inliers_text}, inliers selectionnes visibles {selected_inlier_count}, "
            f"essais {trials_text}, convergence {converged_text}. "
            f"Points finis traces sur toutes les composantes: {finite_count}."
        )
        return _sawada_ransac_figure(
            model,
            source_index,
            color_mode,
        ), caption

    @app.callback(
        Output("spectrogram-stft-caption", "children"),
        Input("split-dropdown", "value"),
        Input("scene-dropdown", "value"),
        Input("algorithm-dropdown", "value"),
        Input("spectrogram-stft-mode-dropdown", "value"),
        Input("nperseg-dropdown", "value"),
    )
    def update_spectrogram_stft_caption(
        split: str,
        scene_id: str,
        algorithm: str,
        stft_mode: str,
        manual_nperseg: int,
    ) -> str:
        if not split or not scene_id or not algorithm:
            return ""

        if (stft_mode or "benchmark") != "benchmark":
            nperseg = int(manual_nperseg or 2048)
            noverlap = nperseg // 2
            return (
                "Spectrogrammes A/B: les traces Sources estimees Sawada affichent "
                "STFT(melange) x masque. Les autres traces utilisent une STFT manuelle: "
                f"fenetre {nperseg}, overlap {noverlap}, nfft auto, window hann."
            )

        bundle = _bundle(config, split, scene_id, algorithm)
        stft_kwargs = _stft_kwargs_from_metrics(
            bundle["metrics"],
            bundle["sawada_model"],
            int(bundle["fs"] or 0),
        )
        nperseg = stft_kwargs.get("nperseg")
        noverlap = stft_kwargs.get("noverlap")
        nfft = stft_kwargs.get("nfft")
        window = stft_kwargs.get("window")
        return (
            "Spectrogrammes A/B: les traces Sources estimees Sawada affichent "
            "STFT(melange) x masque. Les autres traces utilisent la STFT benchmark: "
            f"fenetre {nperseg}, overlap {noverlap}, "
            f"nfft {'auto' if nfft is None else nfft}, window {window}."
        )

    @app.callback(
        Output("overview", "children"),
        Output("warning-slot", "children"),
        Output("time-graph", "figure"),
        Output("spectrogram-graph", "figure"),
        Output("compare-time-graph", "figure"),
        Output("compare-spectrogram-graph", "figure"),
        Output("audio-player", "src"),
        Output("audio-caption", "children"),
        Output("tdoa-graph", "figure"),
        Output("tdoa-table", "data"),
        Input("split-dropdown", "value"),
        Input("scene-dropdown", "value"),
        Input("algorithm-dropdown", "value"),
        Input("group-dropdown", "value"),
        Input("trace-dropdown", "value"),
        Input("compare-group-dropdown", "value"),
        Input("compare-trace-dropdown", "value"),
        Input("spectrogram-stft-mode-dropdown", "value"),
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
        compare_group_key: str,
        compare_trace_index: int,
        spectrogram_stft_mode: str,
        nperseg: int,
        max_frequency: float | None,
        frequency_scale: str,
        spectrogram_scale: str,
    ) -> tuple[
        Any,
        Any,
        go.Figure,
        go.Figure,
        go.Figure,
        go.Figure,
        str,
        str,
        go.Figure,
        list[dict[str, Any]],
    ]:
        if not split or not scene_id or not algorithm or not group_key:
            empty = go.Figure()
            return [], "", empty, empty, empty, empty, "", "", empty, []

        bundle = _bundle(config, split, scene_id, algorithm)
        groups = _signal_groups(bundle)
        if group_key not in groups:
            group_key = next(iter(groups), "")
        if not group_key:
            empty = go.Figure()
            return _overview_children(bundle), "", empty, empty, empty, empty, "", "", empty, []
        group = groups[group_key]
        trace_index = int(trace_index or 0)
        trace_index = min(max(trace_index, 0), len(group.traces) - 1)
        signal = group.data[trace_index]
        duration = len(signal) / group.fs if group.fs else 0.0

        if compare_group_key not in groups:
            compare_group_key = "estimated" if "estimated" in groups else group_key
        compare_group = groups[compare_group_key]
        compare_trace_index = int(compare_trace_index or 0)
        compare_trace_index = min(
            max(compare_trace_index, 0),
            len(compare_group.traces) - 1,
        )

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
        stft_kwargs = (
            _stft_kwargs_from_metrics(bundle["metrics"], bundle["sawada_model"], group.fs)
            if (spectrogram_stft_mode or "benchmark") == "benchmark"
            else None
        )
        return (
            _overview_children(bundle),
            warning,
            _time_figure(group, trace_index),
            _comparison_spectrogram_figure(
                bundle,
                group_key,
                group,
                trace_index,
                nperseg,
                max_frequency,
                frequency_scale,
                spectrogram_scale,
                stft_kwargs,
            ),
            _time_figure(compare_group, compare_trace_index),
            _comparison_spectrogram_figure(
                bundle,
                compare_group_key,
                compare_group,
                compare_trace_index,
                nperseg,
                max_frequency,
                frequency_scale,
                spectrogram_scale,
                stft_kwargs,
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
