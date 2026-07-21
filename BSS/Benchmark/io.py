from __future__ import annotations

import json
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..Dataset import load_scene


DEFAULT_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SceneRecord:
    """Reference legere vers une scene du dataset."""

    scene_id: str
    split: str
    path: Path
    seed: int | None = None
    metadata: dict[str, Any] | None = None


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
