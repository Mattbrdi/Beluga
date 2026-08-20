from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal

try:
    from .io import load_scene, read_manifest
except ImportError:  # pragma: no cover - utile si le fichier est lance directement
    from BSS.Benchmark.io import load_scene, read_manifest


COLUMN_TITLES = (
    "Source 1",
    "Source 2",
    "Melange bruite",
    "Separee 1",
    "Separee 2",
)

SpectrogramData = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz_array(path: Path, key: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if key not in payload:
            raise KeyError(f"Cle '{key}' absente de {path}")
        return np.asarray(payload[key]).copy()


def _load_sawada_model(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]).copy() for key in payload.files}


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


def _clip_stft_kwargs_for_trace(kwargs: dict[str, Any], trace: np.ndarray) -> dict[str, Any]:
    clipped = dict(kwargs)
    n_samples = int(np.asarray(trace).shape[-1])
    if n_samples <= 0:
        return clipped

    nperseg = int(clipped.get("nperseg") or min(256, n_samples))
    if nperseg > n_samples:
        nperseg = n_samples
        clipped["nperseg"] = nperseg

    noverlap = clipped.get("noverlap")
    if noverlap is not None:
        clipped["noverlap"] = min(int(noverlap), max(nperseg - 1, 0))
    return clipped


def _estimated_trace(
    estimated_sources: np.ndarray,
    source_index: int,
    microphone_index: int,
) -> np.ndarray:
    if estimated_sources.ndim == 3:
        if source_index >= estimated_sources.shape[0]:
            raise IndexError("Source estimee absente")
        mic = min(microphone_index, estimated_sources.shape[1] - 1)
        return np.asarray(estimated_sources[source_index, mic], dtype=float)
    if estimated_sources.ndim == 2:
        if source_index >= estimated_sources.shape[0]:
            raise IndexError("Source estimee absente")
        return np.asarray(estimated_sources[source_index], dtype=float)
    if source_index != 0:
        raise IndexError("Source estimee absente")
    return np.asarray(estimated_sources).reshape(-1).astype(float)


def _spectrogram_values(
    trace: np.ndarray,
    stft_kwargs: dict[str, Any],
    spectrogram_scale: str,
    frequency_scale: str,
    max_frequency: float | None,
) -> SpectrogramData:
    kwargs = _clip_stft_kwargs_for_trace(stft_kwargs, trace)
    freqs, times, stft_values = sp_signal.stft(np.asarray(trace, dtype=float), **kwargs)
    values = np.abs(stft_values) ** 2
    if spectrogram_scale == "db":
        values = 10.0 * np.log10(values + 1e-20)

    if max_frequency is not None and max_frequency > 0:
        keep = freqs <= max_frequency
        freqs = freqs[keep]
        values = values[keep]

    if frequency_scale == "log":
        keep = freqs > 0
        freqs = freqs[keep]
        values = values[keep]

    return freqs, times, values, values


def _masked_mixture_spectrogram_values(
    sawada_model: dict[str, np.ndarray],
    source_index: int,
    spectrogram_scale: str,
    frequency_scale: str,
    max_frequency: float | None,
) -> SpectrogramData | None:
    masks = np.asarray(sawada_model.get("masks", []), dtype=float)
    tf_energy = np.asarray(sawada_model.get("tf_energy", []), dtype=float)
    frequencies = np.asarray(sawada_model.get("frequencies", []), dtype=float)
    times = np.asarray(sawada_model.get("times", []), dtype=float)
    if masks.ndim != 3 or tf_energy.ndim != 2 or masks.size == 0 or tf_energy.size == 0:
        return None
    if source_index < 0 or source_index >= masks.shape[0]:
        return None

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
        keep = frequencies <= max_frequency
        frequencies = frequencies[keep]
        energy = energy[keep]
        scale_energy = scale_energy[keep]

    if frequency_scale == "log":
        keep = frequencies > 0
        frequencies = frequencies[keep]
        energy = energy[keep]
        scale_energy = scale_energy[keep]

    if spectrogram_scale == "db":
        values = 10.0 * np.log10(energy + 1e-20)
        limit_values = 10.0 * np.log10(scale_energy + 1e-20)
    else:
        values = energy
        limit_values = scale_energy
    return frequencies, times, values, limit_values


def _panel_limits(values: np.ndarray, spectrogram_scale: str) -> tuple[float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return (-100.0, 0.0) if spectrogram_scale == "db" else (0.0, 1.0)

    if spectrogram_scale == "db":
        vmax = float(np.nanpercentile(finite, 99))
        vmin = max(float(np.nanpercentile(finite, 5)), vmax - 100.0)
    else:
        vmin = float(np.nanpercentile(finite, 1))
        vmax = float(np.nanpercentile(finite, 99))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _global_limits(
    spectrograms: list[list[SpectrogramData]],
    spectrogram_scale: str,
) -> tuple[float, float]:
    values = [
        item[3].ravel()
        for row in spectrograms
        for item in row
        if item[3].size
    ]
    if not values:
        return _panel_limits(np.empty((0,)), spectrogram_scale)
    return _panel_limits(np.concatenate(values), spectrogram_scale)


def build_figure(
    dataset_root: Path,
    results_root: Path,
    split: str,
    algorithm: str,
    max_scenes: int,
    microphone: int,
    frequency_scale: str,
    spectrogram_scale: str,
    max_frequency: float | None,
    estimated_spectrogram_mode: str,
    shared_color_scale: bool,
    fig_width: float,
    row_height: float,
) -> plt.Figure:
    records = read_manifest(dataset_root, split)[:max_scenes]
    if not records:
        raise ValueError(f"Aucune scene trouvee dans {dataset_root / split}")

    scene_ids: list[str] = []
    spectrograms: list[list[SpectrogramData]] = []
    microphone_index = max(0, microphone - 1)

    for record in records:
        scene_dir = results_root / split / record.scene_id
        metrics_path = scene_dir / f"{algorithm}_metrics.json"
        sources_path = scene_dir / f"{algorithm}_sources.npz"
        sawada_model_path = scene_dir / "sawada_model.npz"

        metrics = _read_json(metrics_path)
        sawada_model = (
            _load_sawada_model(sawada_model_path) if algorithm == "sawada" else {}
        )
        estimated_sources = _load_npz_array(sources_path, "sources")
        scene = load_scene(record.path)
        if scene.sources.data.shape[0] < 2:
            raise ValueError(f"Il faut au moins deux sources dans {record.path}")

        fs = int(scene.metadata.fs)
        stft_kwargs = _stft_kwargs_from_metrics(metrics, sawada_model, fs)
        scene_ids.append(record.scene_id)
        row_spectrograms = [
            _spectrogram_values(
                np.asarray(scene.sources.data[0], dtype=float),
                stft_kwargs,
                spectrogram_scale,
                frequency_scale,
                max_frequency,
            ),
            _spectrogram_values(
                np.asarray(scene.sources.data[1], dtype=float),
                stft_kwargs,
                spectrogram_scale,
                frequency_scale,
                max_frequency,
            ),
            _spectrogram_values(
                np.asarray(
                    scene.mixed.data[
                        min(microphone_index, scene.mixed.data.shape[0] - 1)
                    ],
                    dtype=float,
                ),
                stft_kwargs,
                spectrogram_scale,
                frequency_scale,
                max_frequency,
            ),
        ]
        for source_index in (0, 1):
            masked = None
            if estimated_spectrogram_mode == "masked-mixture":
                masked = _masked_mixture_spectrogram_values(
                    sawada_model,
                    source_index,
                    spectrogram_scale,
                    frequency_scale,
                    max_frequency,
                )
            row_spectrograms.append(
                masked
                or _spectrogram_values(
                    _estimated_trace(estimated_sources, source_index, microphone_index),
                    stft_kwargs,
                    spectrogram_scale,
                    frequency_scale,
                    max_frequency,
                )
            )
        spectrograms.append(row_spectrograms)

    n_rows = len(spectrograms)
    n_cols = len(COLUMN_TITLES)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, max(row_height * n_rows, 3.0)),
        squeeze=False,
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#f7f7f3")
    global_limits = (
        _global_limits(spectrograms, spectrogram_scale) if shared_color_scale else None
    )
    last_mesh = None

    for row_index, row in enumerate(spectrograms):
        for col_index, (freqs, times, values, limit_values) in enumerate(row):
            ax = axes[row_index, col_index]
            ax.set_facecolor("#ffffff")
            vmin, vmax = global_limits or _panel_limits(limit_values, spectrogram_scale)
            last_mesh = ax.pcolormesh(
                times,
                freqs,
                values,
                cmap="magma",
                shading="auto",
                vmin=vmin,
                vmax=vmax,
            )
            if frequency_scale == "log":
                ax.set_yscale("log")
            if row_index == 0:
                ax.set_title(COLUMN_TITLES[col_index], fontsize=10)
            if col_index == 0:
                ax.set_ylabel(f"{scene_ids[row_index]}\nHz", fontsize=8)
            else:
                ax.set_yticklabels([])
            if row_index == n_rows - 1:
                ax.set_xlabel("Temps (s)", fontsize=8)
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis="both", labelsize=7, length=2)

    color_label = "Energie (dB)" if spectrogram_scale == "db" else "Energie"
    if last_mesh is not None and shared_color_scale:
        fig.colorbar(
            last_mesh,
            ax=axes.ravel().tolist(),
            shrink=0.65,
            label=color_label,
        )

    title = (
        f"Spectrogrammes benchmark - split={split}, algo={algorithm}, "
        f"micro=M{microphone}"
    )
    fig.suptitle(title, fontsize=13)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genere une grille de spectrogrammes: une ligne par scene du split train "
            "et les colonnes source 1, source 2, melange bruite, separee 1, separee 2."
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
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Chemin PNG/PDF/SVG de sortie.",
    )
    parser.add_argument("--split", default="train", help="Split a afficher, par defaut: train.")
    parser.add_argument(
        "--algorithm",
        default="sawada",
        help="Algorithme benchmark, par defaut: sawada.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=15,
        help="Nombre maximal de scenes/lignes.",
    )
    parser.add_argument(
        "--microphone",
        type=int,
        default=1,
        help="Microphone utilise pour melange et sorties multicanales, en base 1.",
    )
    parser.add_argument("--frequency-scale", choices=("log", "linear"), default="linear")
    parser.add_argument("--spectrogram-scale", choices=("db", "linear"), default="db")
    parser.add_argument(
        "--max-frequency",
        type=float,
        default=12000,
        help="Frequence max affichee en Hz.",
    )
    parser.add_argument(
        "--estimated-spectrogram-mode",
        choices=("masked-mixture", "waveform"),
        default="masked-mixture",
        help=(
            "Pour les sources separees: 'masked-mixture' reproduit la visualisation "
            "Sawada STFT(melange) x masque; 'waveform' recalcule la STFT du signal."
        ),
    )
    parser.add_argument(
        "--shared-color-scale",
        action="store_true",
        help="Utilise la meme echelle couleur pour toutes les cases.",
    )
    parser.add_argument("--fig-width", type=float, default=18.0)
    parser.add_argument("--row-height", type=float, default=1.35)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    if output is None:
        output = args.results / f"{args.split}_{args.algorithm}_spectrogram_grid.png"

    fig = build_figure(
        dataset_root=args.dataset,
        results_root=args.results,
        split=args.split,
        algorithm=args.algorithm,
        max_scenes=args.max_scenes,
        microphone=args.microphone,
        frequency_scale=args.frequency_scale,
        spectrogram_scale=args.spectrogram_scale,
        max_frequency=args.max_frequency,
        estimated_spectrogram_mode=args.estimated_spectrogram_mode,
        shared_color_scale=args.shared_color_scale,
        fig_width=args.fig_width,
        row_height=args.row_height,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Figure sauvegardee: {output}")


if __name__ == "__main__":
    main()
