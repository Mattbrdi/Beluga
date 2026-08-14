from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any, Literal

import numpy as np

from ..Utils.signal_class import Mixture


MetricName = Literal["rmse", "mae"]


@dataclass(frozen=True)
class SourceAlignment:
    """Resultat d'alignement entre sources estimees et sources de reference."""

    aligned_estimated: np.ndarray
    permutation: tuple[int, ...]
    score: float
    metric: MetricName


def true_reference_tdoas_samples(
    scene: Any,
    reference_microphone: int = 0,
) -> np.ndarray:
    """
    Renvoie les TDOA de reference sous forme (n_sources, n_mics), en samples.

    Convention : T[source, mic] = delay(mic) - delay(reference_microphone).
    """
    delay_matrix = np.asarray(scene.metadata.delay_matrix)
    if delay_matrix.ndim != 2:
        raise ValueError("scene.metadata.delay_matrix doit etre une matrice 2D.")
    if not 0 <= reference_microphone < delay_matrix.shape[0]:
        raise ValueError("reference_microphone est hors limites.")
    return (delay_matrix - delay_matrix[reference_microphone, :]).T


def true_reference_tdoas_seconds(
    scene: Any,
    reference_microphone: int = 0,
) -> np.ndarray:
    return true_reference_tdoas_samples(scene, reference_microphone) / scene.metadata.fs


def true_pairwise_tdoas_samples(scene: Any) -> np.ndarray:
    """
    Renvoie les TDOA de reference pairwise, en samples.

    Pour quatre micros, les colonnes sont [M1M2, M1M3, M1M4, M2M3, M2M4, M3M4].
    """
    return Mixture.delay_matrix_to_pairwise_tdoas(scene.metadata.delay_matrix)


def true_pairwise_tdoas_seconds(scene: Any) -> np.ndarray:
    return true_pairwise_tdoas_samples(scene) / scene.metadata.fs


def pairwise_tdoa_labels(n_mics: int) -> list[str]:
    return Mixture.pairwise_tdoa_labels(n_mics)


def _score(errors: np.ndarray, metric: MetricName) -> float:
    if metric == "rmse":
        return float(np.sqrt(np.mean(errors**2)))
    if metric == "mae":
        return float(np.mean(np.abs(errors)))
    raise ValueError(f"Metrique inconnue: {metric}")


def align_sources_by_tdoa(
    estimated_tdoas: np.ndarray,
    target_tdoas: np.ndarray,
    metric: MetricName = "rmse",
) -> SourceAlignment:
    """
    Aligne les sources estimees sur les sources de reference.

    La permutation retournee mappe chaque source de reference vers l'indice de
    source estimee retenu. Autrement dit :
    aligned_estimated = estimated_tdoas[list(permutation)].
    """
    estimated = np.asarray(estimated_tdoas, dtype=float)
    target = np.asarray(target_tdoas, dtype=float)
    if estimated.shape != target.shape:
        raise ValueError(
            f"Shapes TDOA incompatibles: estimated {estimated.shape}, target {target.shape}."
        )

    n_sources = target.shape[0]
    best_permutation: tuple[int, ...] | None = None
    best_aligned: np.ndarray | None = None
    best_score = np.inf

    for permutation in permutations(range(n_sources)):
        aligned = estimated[list(permutation), :]
        score = _score(aligned - target, metric)
        if score < best_score:
            best_score = score
            best_permutation = tuple(int(index) for index in permutation)
            best_aligned = aligned.copy()

    assert best_permutation is not None and best_aligned is not None
    return SourceAlignment(
        aligned_estimated=best_aligned,
        permutation=best_permutation,
        score=best_score,
        metric=metric,
    )


def compute_tdoa_error_metrics(
    aligned_estimated_seconds: np.ndarray,
    target_seconds: np.ndarray,
    fs: float,
) -> dict[str, float]:
    errors_seconds = np.asarray(aligned_estimated_seconds) - np.asarray(target_seconds)
    errors_samples = errors_seconds * fs
    return {
        "mae_seconds": float(np.mean(np.abs(errors_seconds))),
        "rmse_seconds": float(np.sqrt(np.mean(errors_seconds**2))),
        "max_abs_error_seconds": float(np.max(np.abs(errors_seconds))),
        "mae_samples": float(np.mean(np.abs(errors_samples))),
        "rmse_samples": float(np.sqrt(np.mean(errors_samples**2))),
        "max_abs_error_samples": float(np.max(np.abs(errors_samples))),
    }
