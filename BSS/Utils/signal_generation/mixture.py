"""Generation des matrices de melange convolutif."""
from __future__ import annotations

import numpy as np

from ..signal_class import Mixture

class MixtureGenerator:
    def __init__(self, max_delay: int):
        if max_delay < 0:
            raise ValueError("max_delay doit etre positif ou nul.")
        self.max_delay = int(max_delay)

    def generate(
        self,
        rng: np.random.Generator,
        n_sources: int,
        n_mics: int,
        delay_matrix: np.ndarray | None = None,
    ) -> Mixture:
        if delay_matrix is None:
            delay_matrix = rng.integers(
                0,
                self.max_delay + 1,
                size=(n_mics, n_sources),
            )
        delay_matrix = np.asarray(delay_matrix, dtype=int)
        if delay_matrix.shape != (n_mics, n_sources):
            raise ValueError(
                f"delay_matrix doit etre de shape {(n_mics, n_sources)}, obtenu {delay_matrix.shape}."
            )
        return Mixture.create_delay_mixture(
            E=n_sources,
            S=n_mics,
            L=self.max_delay + 1,
            delay_matrix=delay_matrix,
        )
