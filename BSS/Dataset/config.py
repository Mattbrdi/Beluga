from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GeneratorConfig:
    """Parametres transmis a AudioSceneGenerator."""

    fs: int = 8_000
    scene_duration: float = 3.0
    n_sources: int = 2
    n_mics: int = 3
    max_delay: int = 12
    source_placement_rate: float = 0.75
    local_noise_placement_rate: float = 1.0 / 3.0
    source_gain_range: tuple[float, float] = (0.5, 1.0)
    local_noise_gain_range: tuple[float, float] = (0.02, 0.15)
    continuous_noise_gain_range: tuple[float, float] = (0.05, 0.05)

    def __post_init__(self) -> None:
        if self.fs <= 0:
            raise ValueError("fs doit etre strictement positif.")
        if self.scene_duration <= 0:
            raise ValueError("scene_duration doit etre strictement positive.")
        if self.n_sources <= 0 or self.n_mics <= 0:
            raise ValueError("n_sources et n_mics doivent etre strictement positifs.")
        if self.max_delay < 0:
            raise ValueError("max_delay doit etre positif ou nul.")
        for name, placement_rate in (
            ("source_placement_rate", self.source_placement_rate),
            ("local_noise_placement_rate", self.local_noise_placement_rate),
        ):
            if placement_rate < 0:
                raise ValueError(f"Taux de placement invalide pour {name}.")


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration reproductible d'un dataset et de ses splits."""

    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    splits: dict[str, int] = field(
        default_factory=lambda: {"train": 1_000, "validation": 100, "test": 100}
    )
    base_seed: int = 42
    compressed: bool = True

    def __post_init__(self) -> None:
        if self.base_seed < 0:
            raise ValueError("base_seed doit etre positive ou nulle.")
        if not self.splits:
            raise ValueError("Au moins un split doit etre defini.")
        for split_name, size in self.splits.items():
            if not split_name or Path(split_name).name != split_name:
                raise ValueError(f"Nom de split invalide: {split_name!r}.")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError(f"La taille du split {split_name!r} doit etre un entier positif.")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DatasetConfig":
        generator_raw = dict(raw.get("generator", {}))
        for key in (
            "source_gain_range",
            "local_noise_gain_range",
            "continuous_noise_gain_range",
        ):
            if key in generator_raw:
                generator_raw[key] = tuple(generator_raw[key])
        return cls(
            generator=GeneratorConfig(**generator_raw),
            splits=dict(raw.get("splits", {"train": 1_000, "validation": 100, "test": 100})),
            base_seed=int(raw.get("base_seed", 42)),
            compressed=bool(raw.get("compressed", True)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "DatasetConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("La racine de la configuration doit etre un objet JSON.")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
