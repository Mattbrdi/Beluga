from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from itertools import permutations
from pathlib import Path
from typing import Literal

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PIPELINE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from BSS.Algo_Separation.Sawada import SawadaBSS
from BSS.SourceCount import estimate_num_sources
from BSS.Utils.associated_dataclasses import EMClusteringParameters, StftParameters
from BSS.Utils.signal_class import MultiSignal
from src.utils.sub_classes import AudioArray, Environment


GlobalSourceStrategy = Literal["min", "median", "quantile"]


def _default_source_count_stft() -> StftParameters:
    return StftParameters(nperseg=4096, noverlap=3072)


def _default_sawada_stft() -> StftParameters:
    return StftParameters(nperseg=4096, noverlap=3072)


def _default_sawada_em() -> EMClusteringParameters:
    return EMClusteringParameters(
        n_iter=20,
        phi=1.0,
        eps=1e-12,
        energy_threshold_db_above_floor=6.0,
        energy_floor_percentile=20.0,
        min_active_frames_per_frequency=3,
        source_alignment_method="ransac",
        ransac_residual_threshold=0.4,
        ransac_max_trials=300,
        ransac_slope_bound=None,
        ransac_random_state=0,
        ransac_local_optimization_steps=1,
        ransac_slope_grid_size=60,
        ransac_n_local_refinements=2,
        ransac_max_hypotheses_per_pair=100,
    )


@dataclass(frozen=True)
class SourceCountGateConfig:
    """Configuration for per-tetra source-count estimation."""

    enabled: bool = True
    stft_parameters: StftParameters = field(default_factory=_default_source_count_stft)
    method: str = "explained_variance"
    aggregation: str = "quantile"
    aggregation_quantile: float = 0.8
    min_selected_frames: int = 20
    relative_threshold: float = 0.05
    min_eigengap_ratio: float = 3.0
    explained_variance_threshold: float = 0.9
    mask_mode: str = "energy"
    energy_floor_percentile: float = 20.0
    energy_threshold_db_above_floor: float = 6.0
    min_active_run_length: int = 3
    max_frequency: float | None = None
    min_valid_frequencies: int = 2
    min_active_bin_ratio: float = 1e-4
    eps: float = 1e-12


@dataclass(frozen=True)
class SawadaGateConfig:
    """Configuration for deciding and running Sawada."""

    enabled: bool = True
    min_sources_for_separation: int = 2
    min_reliable_tetrahedra: int = 2
    global_source_strategy: GlobalSourceStrategy = "min"
    global_source_quantile: float = 0.5
    require_environment_for_audio_arrays: bool = True
    require_all_tetrahedra_separated: bool = True
    align_sources_across_tetrahedra: bool = True
    stft_parameters: StftParameters = field(default_factory=_default_sawada_stft)
    em_clustering_parameters: EMClusteringParameters = field(default_factory=_default_sawada_em)


@dataclass(frozen=True)
class SourceCountPerTetra:
    tetra_id: str
    estimated_n_sources: int | None
    reliable: bool
    valid_frequency_count: int
    active_bin_ratio: float
    n_sources_per_frequency: np.ndarray
    eigenvalues: np.ndarray
    reason: str


@dataclass(frozen=True)
class SourceSeparationDecision:
    should_separate: bool
    global_n_sources: int | None
    source_counts: list[SourceCountPerTetra]
    separated_audio_arrays_by_tetra: dict[str, list[AudioArray]]
    separated_audio_arrays_by_source: list[list[AudioArray]]
    sawada_models: dict[str, SawadaBSS]
    reason: str


def _keep_consecutive_active_runs(active_frames: np.ndarray, min_run_length: int) -> np.ndarray:
    active_frames = np.asarray(active_frames, dtype=bool)
    if min_run_length <= 1 or active_frames.size == 0:
        return active_frames.copy()

    padded = np.r_[False, active_frames, False]
    transitions = np.diff(padded.astype(int))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    filtered = np.zeros_like(active_frames, dtype=bool)
    for start, stop in zip(starts, stops):
        if stop - start >= min_run_length:
            filtered[start:stop] = True
    return filtered


def _filter_consecutive_runs(mask: np.ndarray, min_run_length: int) -> np.ndarray:
    if min_run_length <= 1:
        return np.asarray(mask, dtype=bool).copy()
    filtered = np.zeros_like(mask, dtype=bool)
    for frequency_index in range(mask.shape[0]):
        filtered[frequency_index] = _keep_consecutive_active_runs(
            mask[frequency_index],
            min_run_length,
        )
    return filtered


class MultiTetraSourceSeparationGate:
    """Estimate source count across tetrahedra and optionally run Sawada.

    The class is intentionally an orchestration layer. It does not modify the
    TDOA code directly. It consumes the one-block ``AudioArray`` list already
    produced by the pipeline and returns either a no-op decision or separated
    ``AudioArray`` objects grouped by source and by tetrahedron.
    """

    def __init__(
        self,
        source_count_config: SourceCountGateConfig | None = None,
        sawada_config: SawadaGateConfig | None = None,
    ) -> None:
        self.source_count_config = source_count_config or SourceCountGateConfig()
        self.sawada_config = sawada_config or SawadaGateConfig()

    def process(
        self,
        audio_arrays: list[AudioArray],
        environment: Environment | None = None,
    ) -> SourceSeparationDecision:
        if not audio_arrays:
            return self._no_separation([], "Aucun AudioArray a traiter.")

        source_counts = [self.estimate_tetra_source_count(item) for item in audio_arrays]
        global_n_sources = self._global_source_count(source_counts)
        should_separate, reason = self._should_run_sawada(source_counts, global_n_sources)
        if not should_separate:
            return self._no_separation(source_counts, reason, global_n_sources)

        if environment is None and self.sawada_config.require_environment_for_audio_arrays:
            return self._no_separation(
                source_counts,
                "Environment absent: impossible de reconstruire des AudioArray separes.",
                global_n_sources,
            )

        try:
            by_tetra, models = self._separate_all_tetrahedra(
                audio_arrays,
                int(global_n_sources),
                environment,
            )
        except Exception as exc:
            return self._no_separation(
                source_counts,
                f"Echec Sawada: {exc}",
                global_n_sources,
            )

        if len(by_tetra) != len(audio_arrays) and self.sawada_config.require_all_tetrahedra_separated:
            return self._no_separation(
                source_counts,
                "Toutes les tetraedres n'ont pas pu etre separes.",
                global_n_sources,
            )

        by_source = self._group_by_source(audio_arrays, by_tetra, int(global_n_sources))
        return SourceSeparationDecision(
            should_separate=True,
            global_n_sources=int(global_n_sources),
            source_counts=source_counts,
            separated_audio_arrays_by_tetra=by_tetra,
            separated_audio_arrays_by_source=by_source,
            sawada_models=models,
            reason=reason,
        )

    def estimate_tetra_source_count(self, audio_array: AudioArray) -> SourceCountPerTetra:
        frequencies, _, X = self._stft(audio_array)
        mask = self._activity_mask(X, frequencies)
        config = self.source_count_config
        result = estimate_num_sources(
            X,
            mask,
            method=config.method,
            min_selected_frames=config.min_selected_frames,
            relative_threshold=config.relative_threshold,
            min_eigengap_ratio=config.min_eigengap_ratio,
            explained_variance_threshold=config.explained_variance_threshold,
            aggregation=config.aggregation,
            aggregation_quantile=config.aggregation_quantile,
            eps=config.eps,
        )
        estimated = result["estimated_n_sources"]
        valid_frequency_count = int(
            np.sum(np.asarray(result["valid_frequencies"], dtype=bool))
        )
        active_bin_ratio = float(np.mean(mask))
        reliable = (
            estimated is not None
            and valid_frequency_count >= config.min_valid_frequencies
            and active_bin_ratio >= config.min_active_bin_ratio
        )
        reason = "ok" if reliable else "estimation non fiable"
        return SourceCountPerTetra(
            tetra_id=audio_array.metadata.tetra_id,
            estimated_n_sources=None if estimated is None else int(estimated),
            reliable=reliable,
            valid_frequency_count=valid_frequency_count,
            active_bin_ratio=active_bin_ratio,
            n_sources_per_frequency=np.asarray(
                result["n_sources_per_frequency"],
                dtype=float,
            ),
            eigenvalues=np.asarray(result["eigenvalues"], dtype=float),
            reason=reason,
        )

    def _stft(self, audio_array: AudioArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from scipy import signal as sp_signal

        params = self.source_count_config.stft_parameters
        frequencies, times, stft_values = sp_signal.stft(
            np.asarray(audio_array.data_array, dtype=float),
            fs=audio_array.metadata.sample_rate,
            window=params.window,
            nperseg=params.nperseg,
            noverlap=params.noverlap,
            nfft=params.nfft,
            boundary=params.boundary,
            padded=params.padded,
            axis=-1,
        )
        X = np.moveaxis(stft_values, 0, -1)
        return frequencies, times, X

    def _activity_mask(self, X: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
        config = self.source_count_config
        if config.mask_mode == "all":
            mask = np.ones(X.shape[:2], dtype=bool)
        elif config.mask_mode == "energy":
            energy = np.sum(np.abs(X) ** 2, axis=2)
            energy_db = 10.0 * np.log10(energy + config.eps)
            floor_db = float(np.percentile(energy_db, config.energy_floor_percentile))
            threshold = floor_db + config.energy_threshold_db_above_floor
            mask = energy_db >= threshold
        else:
            raise ValueError(f"Mode de masque inconnu: {config.mask_mode!r}")

        if config.max_frequency is not None and config.max_frequency > 0:
            mask = mask.copy()
            mask[frequencies > config.max_frequency, :] = False
        return _filter_consecutive_runs(mask, config.min_active_run_length)

    def _global_source_count(self, source_counts: list[SourceCountPerTetra]) -> int | None:
        reliable_counts = np.asarray(
            [
                item.estimated_n_sources
                for item in source_counts
                if item.reliable and item.estimated_n_sources is not None
            ],
            dtype=float,
        )
        if reliable_counts.size == 0:
            return None

        strategy = self.sawada_config.global_source_strategy
        if strategy == "min":
            return int(np.min(reliable_counts))
        if strategy == "median":
            return int(round(float(np.median(reliable_counts))))
        if strategy == "quantile":
            q = min(max(float(self.sawada_config.global_source_quantile), 0.0), 1.0)
            return int(np.ceil(float(np.quantile(reliable_counts, q))))
        raise ValueError(f"Strategie globale inconnue: {strategy!r}")

    def _should_run_sawada(
        self,
        source_counts: list[SourceCountPerTetra],
        global_n_sources: int | None,
    ) -> tuple[bool, str]:
        if not self.source_count_config.enabled:
            return False, "Estimation du nombre de sources desactivee."
        if not self.sawada_config.enabled:
            return False, "Sawada desactive."

        reliable_count = sum(item.reliable for item in source_counts)
        if reliable_count < self.sawada_config.min_reliable_tetrahedra:
            return (
                False,
                f"Pas assez de tetraedres fiables ({reliable_count}/"
                f"{self.sawada_config.min_reliable_tetrahedra}).",
            )
        if global_n_sources is None:
            return False, "Nombre global de sources indetermine."
        if global_n_sources < self.sawada_config.min_sources_for_separation:
            return (
                False,
                f"Nombre global de sources {global_n_sources}: separation inutile.",
            )
        return True, f"Sawada lance avec {global_n_sources} sources."

    def _separate_all_tetrahedra(
        self,
        audio_arrays: list[AudioArray],
        n_sources: int,
        environment: Environment | None,
    ) -> tuple[dict[str, list[AudioArray]], dict[str, SawadaBSS]]:
        by_tetra: dict[str, list[AudioArray]] = {}
        models: dict[str, SawadaBSS] = {}
        reference_envelopes: list[np.ndarray] | None = None

        for audio_array in audio_arrays:
            tetra_id = audio_array.metadata.tetra_id
            model = SawadaBSS(
                n_sources=n_sources,
                stft_parameters=self.sawada_config.stft_parameters,
                em_clustering_parameters=self.sawada_config.em_clustering_parameters,
            )
            model.process_signal(
                MultiSignal.from_array(
                    audio_array.data_array,
                    audio_array.metadata.sample_rate,
                )
            )
            separated = model.separate_source()
            if self.sawada_config.align_sources_across_tetrahedra:
                envelopes = [self._source_envelope(item.data) for item in separated]
                if reference_envelopes is None:
                    reference_envelopes = envelopes
                else:
                    order = self._best_envelope_permutation(reference_envelopes, envelopes)
                    separated = [separated[index] for index in order]

            by_tetra[tetra_id] = [
                self._to_audio_array(source, audio_array, environment)
                for source in separated
            ]
            models[tetra_id] = model

        return by_tetra, models

    def _to_audio_array(
        self,
        source_signal: MultiSignal,
        template: AudioArray,
        environment: Environment | None,
    ) -> AudioArray:
        if environment is None:
            raise ValueError("Environment requis pour reconstruire un AudioArray.")
        tetra_id = template.metadata.tetra_id
        if tetra_id not in environment.tetrahedras:
            raise KeyError(f"Tetraedre {tetra_id!r} absent de l'environnement.")
        metadata = replace(template.metadata)
        return AudioArray(
            metadata,
            environment.tetrahedras[tetra_id],
            template.use_h4,
            data_array=source_signal.data,
        )

    def _group_by_source(
        self,
        original_audio_arrays: list[AudioArray],
        separated_by_tetra: dict[str, list[AudioArray]],
        n_sources: int,
    ) -> list[list[AudioArray]]:
        grouped: list[list[AudioArray]] = [[] for _ in range(n_sources)]
        for audio_array in original_audio_arrays:
            separated = separated_by_tetra.get(audio_array.metadata.tetra_id, [])
            for source_index in range(min(n_sources, len(separated))):
                grouped[source_index].append(separated[source_index])
        return grouped

    @staticmethod
    def _source_envelope(data: np.ndarray) -> np.ndarray:
        energy = np.mean(np.asarray(data, dtype=float) ** 2, axis=0)
        energy = energy - float(np.mean(energy))
        norm = float(np.linalg.norm(energy))
        if norm <= 1e-12:
            return np.zeros_like(energy)
        return energy / norm

    @staticmethod
    def _best_envelope_permutation(
        reference_envelopes: list[np.ndarray],
        candidate_envelopes: list[np.ndarray],
    ) -> tuple[int, ...]:
        n_sources = min(len(reference_envelopes), len(candidate_envelopes))
        best_order = tuple(range(n_sources))
        best_score = -np.inf
        for order in permutations(range(n_sources)):
            score = 0.0
            for reference_index, candidate_index in enumerate(order):
                ref = reference_envelopes[reference_index]
                cand = candidate_envelopes[candidate_index]
                n = min(ref.size, cand.size)
                if n:
                    score += float(ref[:n] @ cand[:n])
            if score > best_score:
                best_score = score
                best_order = tuple(int(index) for index in order)
        return best_order

    @staticmethod
    def _no_separation(
        source_counts: list[SourceCountPerTetra],
        reason: str,
        global_n_sources: int | None = None,
    ) -> SourceSeparationDecision:
        return SourceSeparationDecision(
            should_separate=False,
            global_n_sources=global_n_sources,
            source_counts=source_counts,
            separated_audio_arrays_by_tetra={},
            separated_audio_arrays_by_source=[],
            sawada_models={},
            reason=reason,
        )
