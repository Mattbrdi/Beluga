from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset: Path
    split: str
    output: Path
    algorithms: tuple[str, ...] = ("sawada", "ica")
    reference_microphone: int = 0
    limit: int | None = None
    sawada_min_frequency_hz: float | None = None
    sawada_max_frequency_hz: float | None = None

    def __post_init__(self) -> None:
        if not self.algorithms:
            raise ValueError("Au moins un algorithme doit etre demande.")
        unknown = set(self.algorithms) - {"sawada", "ica"}
        if unknown:
            raise ValueError(f"Algorithmes inconnus: {sorted(unknown)}")
        if self.reference_microphone < 0:
            raise ValueError("reference_microphone doit etre positif ou nul.")
        if self.limit is not None and self.limit < 0:
            raise ValueError("limit doit etre positif ou nul.")
        if (
            self.sawada_min_frequency_hz is not None
            and self.sawada_max_frequency_hz is not None
            and self.sawada_min_frequency_hz >= self.sawada_max_frequency_hz
        ):
            raise ValueError(
                "sawada_min_frequency_hz doit etre inferieur a "
                "sawada_max_frequency_hz."
            )
