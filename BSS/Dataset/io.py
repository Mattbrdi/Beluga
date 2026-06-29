from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..Utils.signal_class import Mixture, MultiSignal
from ..Utils.signal_generation import (
    AudioScene,
    AudioSceneMetadata,
    CompositeSignalMetadata,
    ContinuousNoiseMetadata,
    SignalPlacementMetadata,
)


FORMAT_VERSION = 1
SCENE_ARRAY_KEYS = (
    "sources",
    "clean_mixed",
    "noise",
    "mixed",
    "mixing_filters",
    "fs",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def save_scene(scene: AudioScene, path: str | Path, compressed: bool = True) -> Path:
    """Sauvegarde toutes les references utiles au benchmark dans un seul NPZ."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    save = np.savez_compressed if compressed else np.savez
    save(
        target,
        sources=scene.sources.data,
        clean_mixed=scene.clean_mixed.data,
        noise=scene.noise.data,
        mixed=scene.mixed.data,
        mixing_filters=scene.mixing.filters,
        fs=np.asarray(scene.metadata.fs),
        metadata_json=np.asarray(
            json.dumps(metadata_to_dict(scene), ensure_ascii=False)
        ),
        format_version=np.asarray(FORMAT_VERSION),
    )
    return target


def load_scene_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Charge uniquement les tableaux NumPy, sans reconstruire les objets BSS."""
    with np.load(Path(path), allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]).copy() for key in SCENE_ARRAY_KEYS}


def load_scene(path: str | Path) -> AudioScene:
    """Reconstruit une AudioScene directement exploitable par les algorithmes BSS."""
    target = Path(path)
    with np.load(target, allow_pickle=False) as payload:
        if "format_version" not in payload or "metadata_json" not in payload:
            raise ValueError(
                f"Le fichier {target} utilise l'ancien format sans metadonnees. "
                "Regenere le dataset avec la version actuelle."
            )

        version = int(np.asarray(payload["format_version"]).item())
        if version != FORMAT_VERSION:
            raise ValueError(
                f"Version de scene non supportee: {version}; "
                f"version attendue: {FORMAT_VERSION}."
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

    return AudioScene(
        sources=MultiSignal.from_array(sources_data, fs),
        mixing=mixing,
        clean_mixed=MultiSignal.from_array(clean_mixed_data, fs),
        noise=MultiSignal.from_array(noise_data, fs),
        mixed=MultiSignal.from_array(mixed_data, fs),
        metadata=_metadata_from_dict(metadata_raw),
    )


def _metadata_from_dict(raw: dict[str, Any]) -> AudioSceneMetadata:
    def composite(value: dict[str, Any]) -> CompositeSignalMetadata:
        return CompositeSignalMetadata(
            placements=[
                SignalPlacementMetadata(**placement)
                for placement in value["placements"]
            ],
            duration=float(value["duration"]),
        )

    return AudioSceneMetadata(
        fs=int(raw["fs"]),
        duration=float(raw["duration"]),
        n_sources=int(raw["n_sources"]),
        n_mics=int(raw["n_mics"]),
        source_composites=[composite(value) for value in raw["source_composites"]],
        local_noise_composites=[
            composite(value) for value in raw["local_noise_composites"]
        ],
        continuous_noises=[
            ContinuousNoiseMetadata(**value)
            for value in raw["continuous_noises"]
        ],
        max_delay=int(raw["max_delay"]),
        delay_matrix=np.asarray(raw["delay_matrix"], dtype=int),
        seed=None if raw["seed"] is None else int(raw["seed"]),
    )


def metadata_to_dict(scene: AudioScene) -> dict[str, Any]:
    return _json_value(asdict(scene.metadata))


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(_json_value(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
