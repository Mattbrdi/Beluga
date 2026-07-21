from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

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
