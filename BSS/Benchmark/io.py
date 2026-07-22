from __future__ import annotations

import json
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..Utils.signal_class import Mixture, MultiSignal


DEFAULT_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SceneRecord:
    """Reference legere vers une scene du dataset."""

    scene_id: str
    split: str
    path: Path
    seed: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class BenchmarkSceneMetadata:
    fs: int
    duration: float
    n_sources: int
    n_mics: int
    max_delay: int
    delay_matrix: np.ndarray
    seed: int | None = None
    snr_db: float | None = None


@dataclass(frozen=True)
class BenchmarkScene:
    sources: MultiSignal
    mixing: Mixture
    clean_mixed: MultiSignal
    noise: MultiSignal
    mixed: MultiSignal
    metadata: BenchmarkSceneMetadata


def _metadata_from_dict(raw: dict[str, Any]) -> BenchmarkSceneMetadata:
    return BenchmarkSceneMetadata(
        fs=int(raw["fs"]),
        duration=float(raw["duration"]),
        n_sources=int(raw["n_sources"]),
        n_mics=int(raw["n_mics"]),
        max_delay=int(raw["max_delay"]),
        delay_matrix=np.asarray(raw["delay_matrix"], dtype=int),
        seed=None if raw.get("seed") is None else int(raw["seed"]),
        snr_db=None if raw.get("snr_db") is None else float(raw["snr_db"]),
    )


def load_scene(path: str | Path) -> BenchmarkScene:
    """Charge une scene NPZ sans importer le generateur de dataset complet."""
    target = Path(path)
    with np.load(target, allow_pickle=False) as payload:
        if "format_version" not in payload or "metadata_json" not in payload:
            raise ValueError(
                f"Le fichier {target} utilise l'ancien format sans metadonnees. "
                "Regenere le dataset avec la version actuelle."
            )

        fs = int(np.asarray(payload["fs"]).item())
        sources_data = np.asarray(payload["sources"]).copy()
        clean_mixed_data = np.asarray(payload["clean_mixed"]).copy()
        noise_data = np.asarray(payload["noise"]).copy()
        mixed_data = np.asarray(payload["mixed"]).copy()
        mixing_filters = np.asarray(payload["mixing_filters"]).copy()
        metadata_raw = json.loads(str(np.asarray(payload["metadata_json"]).item()))

    if mixing_filters.ndim != 3:
        raise ValueError(
            "mixing_filters doit avoir la forme (n_mics, n_sources, filter_length)."
        )

    n_mics, n_sources, filter_length = mixing_filters.shape
    mixing = Mixture(E=n_sources, S=n_mics, L=filter_length)
    mixing.filters = mixing_filters

    return BenchmarkScene(
        sources=MultiSignal.from_array(sources_data, fs),
        mixing=mixing,
        clean_mixed=MultiSignal.from_array(clean_mixed_data, fs),
        noise=MultiSignal.from_array(noise_data, fs),
        mixed=MultiSignal.from_array(mixed_data, fs),
        metadata=_metadata_from_dict(metadata_raw),
    )


def read_manifest(dataset_root: str | Path, split: str) -> list[SceneRecord]:
    """Lit le manifest d'un split et renvoie les chemins absolus des scenes."""
    dataset_root = Path(dataset_root)
    split_dir = dataset_root / split
    manifest_path = split_dir / "manifest.jsonl"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest introuvable: {manifest_path}")

    records: list[SceneRecord] = []
    with manifest_path.open(mode="r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            scene_path = split_dir / item["path"]
            if not scene_path.exists():
                raise FileNotFoundError(
                    f"Scene introuvable ligne {line_number}: {scene_path}"
                )

            records.append(
                SceneRecord(
                    scene_id=item["id"],
                    split=item.get("split", split),
                    path=scene_path,
                    seed=item.get("seed"),
                    metadata=item.get("metadata"),
                )
            )

    return records


def iter_scene_records(
    dataset_root: str | Path,
    split: str = "test",
    scene_ids: set[str] | None = None,
) -> Iterator[SceneRecord]:
    """Itere sur les scenes referencees par les manifests, sans les charger."""
    if split == "all":
        for split_name in DEFAULT_SPLITS:
            yield from iter_scene_records(dataset_root, split_name, scene_ids)
        return

    for record in read_manifest(dataset_root, split):
        if scene_ids is not None and record.scene_id not in scene_ids:
            continue
        yield record


def iter_scenes(
    dataset_root: str | Path,
    split: str = "test",
    limit: int | None = None,
    scene_ids: set[str] | None = None,
) -> Iterator[tuple[SceneRecord, Any]]:
    """Charge les scenes une par une pour alimenter le benchmark."""
    for index, record in enumerate(iter_scene_records(dataset_root, split, scene_ids)):
        if limit is not None and index >= limit:
            break

        try:
            scene = load_scene(record.path)
        except Exception as exc:
            raise RuntimeError(
                f"Impossible de charger la scene {record.scene_id}: {record.path}"
            ) from exc

        yield record, scene


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def scene_result_dir(output_root: str | Path, split: str, scene_id: str) -> Path:
    return Path(output_root) / split / scene_id


def sources_to_array(sources: list[Any]) -> np.ndarray:
    """Convertit une liste de MultiSignal en array (n_sources, n_mics, n_samples)."""
    if not sources:
        return np.empty((0, 0, 0), dtype=float)

    n_sources = len(sources)
    n_mics = sources[0].num_signals
    max_length = max(source.data.shape[1] for source in sources)
    array = np.zeros((n_sources, n_mics, max_length), dtype=float)

    for source_index, source in enumerate(sources):
        data = source.data
        if data.shape[0] != n_mics:
            raise ValueError("Toutes les sources doivent avoir le meme nombre de micros.")
        array[source_index, :, : data.shape[1]] = data

    return array


def save_sources_npz(
    path: str | Path,
    result: Any,
    true_tdoas_seconds: np.ndarray,
    true_tdoas_samples: np.ndarray,
    aligned_tdoas_seconds: np.ndarray,
    aligned_tdoas_samples: np.ndarray,
    pairwise_labels: list[str],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fs = 0 if not result.sources else result.sources[0].freq
    np.savez_compressed(
        target,
        sources=sources_to_array(result.sources),
        fs=np.asarray(fs),
        estimated_tdoas_seconds=result.estimated_tdoas_seconds,
        estimated_tdoas_samples=result.estimated_tdoas_samples,
        aligned_tdoas_seconds=aligned_tdoas_seconds,
        aligned_tdoas_samples=aligned_tdoas_samples,
        true_tdoas_seconds=true_tdoas_seconds,
        true_tdoas_samples=true_tdoas_samples,
        pairwise_labels=np.asarray(pairwise_labels),
    )
    return target


def save_sawada_model_npz(path: str | Path, result: Any) -> Path | None:
    """Sauvegarde l'etat final utile de Sawada pour la visualisation/debug."""
    payload = result.debug_artifacts.get("sawada_model", {})
    if not payload:
        return None

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        masks=np.asarray(payload["masks"]),
        posteriors=np.asarray(payload["posteriors"]),
        bin_vectors=np.asarray(payload.get("bin_vectors", np.empty((0, 0, 0)))),
        tf_energy=np.asarray(payload.get("tf_energy", np.empty((0, 0)))),
        frequency_energy=np.asarray(payload.get("frequency_energy", np.empty((0,)))),
        active_tf_mask=np.asarray(payload.get("active_tf_mask", np.empty((0, 0)))),
        energy_threshold_db=np.asarray(payload.get("energy_threshold_db", np.nan)),
        frequencies=np.asarray(payload["frequencies"]),
        times=np.asarray(payload["times"]),
        centroids=np.asarray(payload["centroids"]),
        variances=np.asarray(payload["variances"]),
        weights=np.asarray(payload["weights"]),
        whitening=np.asarray(payload["whitening"]),
    )
    return target


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_json_value(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def write_summary_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return target

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with target.open(mode="w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_value(row.get(key)) for key in fieldnames})

    return target
