from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import permutations
from typing import Dict, List, Optional

import numpy as np
from scipy import signal as sp_signal

from .em_clustering import EMClustering
from .results import SawadaDebugArtifacts, SawadaResult
from .source_assignment import (
    CentroidSourceAssignment,
    fit_centroid_source_trajectories,
    labels_to_source_masks,
)
from ...Utils.associated_dataclasses import (
    EMClusteringParameters,
    SawadaBssParameters,
    StftParameters,
)
from ...Utils.signal_class import MultiSignal, NSpectrogram


def _normalize_for_correlation(signal: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    data = np.asarray(signal, dtype=float)
    data = data - np.mean(data)
    norm = np.linalg.norm(data)
    if norm <= eps:
        return data
    return data / norm


def _estimate_lag_by_correlation(
    reference: np.ndarray,
    target: np.ndarray,
    max_lag_samples: int | None = None,
) -> int:
    """
    Estime delay(target) - delay(reference) par maximum de correlation.
    """
    if max_lag_samples is not None and max_lag_samples < 0:
        raise ValueError("max_lag_samples doit etre positif ou nul.")

    reference_data = _normalize_for_correlation(reference)
    target_data = _normalize_for_correlation(target)
    correlation = sp_signal.correlate(
        target_data,
        reference_data,
        mode="full",
        method="fft",
    )
    lags = sp_signal.correlation_lags(
        target_data.size,
        reference_data.size,
        mode="full",
    )

    if max_lag_samples is not None:
        valid = np.abs(lags) <= max_lag_samples
        correlation = correlation[valid]
        lags = lags[valid]
        if lags.size == 0:
            raise ValueError("Aucun lag valide avec ce max_lag_samples.")

    return int(lags[int(np.argmax(np.abs(correlation)))])


def _pairwise_tdoa_count(n_mics: int) -> int:
    return n_mics * (n_mics - 1) // 2

@dataclass
class SawadaBSS:

    """
    Algorithme de Sawada:

    préprocess:
    - Apllique la stft pour obtenir le NSpectrogram
    - Normalise par bin le Nspectrogram
    algo:
    - clustering
    - alignement des cluster entre les fréquence

    - utilisation des mask obtenue pour obtenir le Nspectrogram pour une source donnée
    - retour dans le domaine temporel

    """
    n_sources: int
    stft_parameters: StftParameters = field(default_factory= StftParameters)
    em_clustering_parameters : EMClusteringParameters = field(default_factory= EMClusteringParameters)

    # État de l'algorithme (Champs calculés plus tard)
    # On utilise default_factory pour les dictionnaires vides
    signal: Optional['MultiSignal'] = None
    nspectro_preprocessed: Optional[NSpectrogram] = None
    bin_models: Dict[int, 'EMClustering'] = field(default_factory=dict)
    bin_masks: Dict[int, np.ndarray] = field(default_factory=dict)
    bin_posteriors: Dict[int, np.ndarray] = field(default_factory=dict)
    bin_active_clusters: Dict[int, np.ndarray] = field(default_factory=dict)
    cluster_masks_before_assignment: Optional[np.ndarray] = None
    cluster_posteriors_before_assignment: Optional[np.ndarray] = None
    cluster_active_before_assignment: Optional[np.ndarray] = None
    tf_energy: Optional[np.ndarray] = None
    active_tf_mask: Optional[np.ndarray] = None
    active_frequency_mask: Optional[np.ndarray] = None
    source_assignment: Optional[CentroidSourceAssignment] = None
    source_assignment_labels: Optional[np.ndarray] = None
    source_assignment_distances: Optional[np.ndarray] = None
    source_assignment_relative_phases: Optional[np.ndarray] = None
    source_assignment_selected_labels: Optional[np.ndarray] = None
    source_assignment_slopes: Optional[np.ndarray] = None
    source_assignment_intercepts: Optional[np.ndarray] = None
    source_assignment_scores: Optional[np.ndarray] = None
    source_assignment_n_inliers: Optional[np.ndarray] = None
    source_assignment_n_trials: Optional[np.ndarray] = None
    source_assignment_converged: Optional[np.ndarray] = None
    source_assignment_frequency_inliers: Optional[np.ndarray] = None
    source_assignment_selected_centroids: Optional[np.ndarray] = None
    energy_threshold_db: Optional[float] = None

    @property
    def parameters(self) -> SawadaBssParameters:
        return SawadaBssParameters(
            n_sources=self.n_sources,
            stft_parameters=self.stft_parameters,
            em_clustering_parameters=self.em_clustering_parameters,
        )

    def _artifact_tf_energy(self) -> np.ndarray:
        if self.tf_energy is not None:
            return self.tf_energy
        if self.signal is not None:
            raw_spectro = self.get_spectro(self.signal)
            return np.sum(np.abs(raw_spectro.Sxx) ** 2, axis=0)
        if self.nspectro_preprocessed is not None:
            return np.sum(np.abs(self.nspectro_preprocessed.Sxx) ** 2, axis=0)
        return np.empty((0, 0), dtype=float)

    def _artifact_em_statistics(self, n_freqs: int) -> tuple[np.ndarray, np.ndarray]:
        variances = np.full((n_freqs, self.n_sources), np.nan, dtype=float)
        weights = np.full((n_freqs, self.n_sources), np.nan, dtype=float)
        for frequency_index, bin_model in self.bin_models.items():
            variances[frequency_index] = bin_model.variances
            weights[frequency_index] = bin_model.weights
        return variances, weights

    def to_debug_artifacts(self) -> SawadaDebugArtifacts:
        """Construit les artefacts lourds destines au benchmark et aux plots."""
        if self.nspectro_preprocessed is None:
            raise RuntimeError("Il faut appeler process_signal avant to_debug_artifacts.")

        tf_energy = self._artifact_tf_energy()
        active_tf_mask = (
            self.active_tf_mask
            if self.active_tf_mask is not None
            else np.ones_like(tf_energy, dtype=bool)
        )
        n_freqs = self.nspectro_preprocessed.Sxx.shape[1]
        variances, weights = self._artifact_em_statistics(n_freqs)

        return SawadaDebugArtifacts(
            masks=self.get_final_masks().astype(np.uint8),
            posteriors=self.get_final_posteriors(),
            active_clusters=self.get_final_active_clusters().astype(np.uint8),
            cluster_masks=np.asarray(
                self.cluster_masks_before_assignment
                if self.cluster_masks_before_assignment is not None
                else np.empty((0, 0, 0))
            ),
            cluster_posteriors=np.asarray(
                self.cluster_posteriors_before_assignment
                if self.cluster_posteriors_before_assignment is not None
                else np.empty((0, 0, 0))
            ),
            cluster_active=np.asarray(
                self.cluster_active_before_assignment
                if self.cluster_active_before_assignment is not None
                else np.empty((0, 0))
            ),
            active_frequency_mask=(
                np.asarray(self.active_frequency_mask, dtype=np.uint8)
                if self.active_frequency_mask is not None
                else np.ones(n_freqs, dtype=np.uint8)
            ),
            bin_vectors=self.nspectro_preprocessed.Sxx,
            tf_energy=tf_energy,
            frequency_energy=np.mean(tf_energy, axis=1),
            active_tf_mask=active_tf_mask.astype(np.uint8),
            energy_threshold_db=np.asarray(
                np.nan if self.energy_threshold_db is None else self.energy_threshold_db
            ),
            frequencies=np.asarray(self.nspectro_preprocessed.f, dtype=float),
            times=np.asarray(self.nspectro_preprocessed.t, dtype=float),
            centroids=self.all_centroids,
            variances=variances,
            weights=weights,
            source_assignment_labels=np.asarray(
                self.source_assignment_labels
                if self.source_assignment_labels is not None
                else np.empty((0, 0))
            ),
            source_assignment_distances=np.asarray(
                self.source_assignment_distances
                if self.source_assignment_distances is not None
                else np.empty((0, 0))
            ),
            source_assignment_relative_phases=np.asarray(
                self.source_assignment_relative_phases
                if self.source_assignment_relative_phases is not None
                else np.empty((0, 0, 0))
            ),
            source_assignment_selected_labels=np.asarray(
                self.source_assignment_selected_labels
                if self.source_assignment_selected_labels is not None
                else np.empty((0, 0))
            ),
            source_assignment_slopes=np.asarray(
                self.source_assignment_slopes
                if self.source_assignment_slopes is not None
                else np.empty((0, 0))
            ),
            source_assignment_intercepts=np.asarray(
                self.source_assignment_intercepts
                if self.source_assignment_intercepts is not None
                else np.empty((0, 0))
            ),
            source_assignment_scores=np.asarray(
                self.source_assignment_scores
                if self.source_assignment_scores is not None
                else np.empty((0,))
            ),
            source_assignment_n_inliers=np.asarray(
                self.source_assignment_n_inliers
                if self.source_assignment_n_inliers is not None
                else np.empty((0,))
            ),
            source_assignment_n_trials=np.asarray(
                self.source_assignment_n_trials
                if self.source_assignment_n_trials is not None
                else np.empty((0,))
            ),
            source_assignment_converged=np.asarray(
                self.source_assignment_converged
                if self.source_assignment_converged is not None
                else np.empty((0,))
            ),
            source_assignment_frequency_inliers=np.asarray(
                self.source_assignment_frequency_inliers
                if self.source_assignment_frequency_inliers is not None
                else np.empty((0, 0))
            ),
            source_assignment_selected_centroids=np.asarray(
                self.source_assignment_selected_centroids
                if self.source_assignment_selected_centroids is not None
                else np.empty((0, 0))
            ),
        )

    def to_result(self, include_diagnostics: bool = False) -> SawadaResult:
        """Retourne une vue structurée des sorties principales de Sawada."""
        diagnostics = self.to_debug_artifacts() if include_diagnostics else None
        if self.nspectro_preprocessed is None:
            raise RuntimeError("Il faut appeler process_signal avant to_result.")

        return SawadaResult(
            masks=(
                diagnostics.masks
                if diagnostics is not None
                else self.get_final_masks().astype(np.uint8)
            ),
            posteriors=(
                diagnostics.posteriors
                if diagnostics is not None
                else self.get_final_posteriors()
            ),
            active_clusters=(
                diagnostics.active_clusters
                if diagnostics is not None
                else self.get_final_active_clusters().astype(np.uint8)
            ),
            centroids=(
                diagnostics.centroids
                if diagnostics is not None
                else self.all_centroids
            ),
            frequencies=np.asarray(self.nspectro_preprocessed.f, dtype=float),
            times=np.asarray(self.nspectro_preprocessed.t, dtype=float),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _keep_consecutive_active_runs(active_frames: np.ndarray, min_run_length: int) -> np.ndarray:
        """
        Conserve uniquement les groupes temporels actifs suffisamment longs.

        Un bin temps-frequence actif isole ne porte pas assez d'information locale
        pour stabiliser l'EM; on le retire donc avant le clustering.
        """
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

    def _filter_active_tf_runs(self, active_mask: np.ndarray) -> np.ndarray:
        min_run_length = max(
            1,
            int(self.em_clustering_parameters.min_active_frames_per_frequency),
        )
        if min_run_length <= 1:
            return np.asarray(active_mask, dtype=bool).copy()

        filtered_mask = np.zeros_like(active_mask, dtype=bool)
        for frequency_index in range(active_mask.shape[0]):
            filtered_mask[frequency_index] = self._keep_consecutive_active_runs(
                active_mask[frequency_index],
                min_run_length,
            )
        return filtered_mask

    def _compute_active_tf_mask(self, tf_energy: np.ndarray) -> np.ndarray:
        threshold_margin = self.em_clustering_parameters.energy_threshold_db_above_floor
        active_mask = np.ones_like(tf_energy, dtype=bool)
        self.energy_threshold_db = None
        if threshold_margin is None:
            return self._filter_active_tf_runs(active_mask)

        energy_db = 10 * np.log10(tf_energy + self.em_clustering_parameters.eps)
        floor_db = np.percentile(
            energy_db,
            self.em_clustering_parameters.energy_floor_percentile,
        )
        self.energy_threshold_db = float(floor_db + threshold_margin)
        active_mask = energy_db >= self.energy_threshold_db
        return self._filter_active_tf_runs(active_mask)



    def preprocess(self, input: MultiSignal) -> NSpectrogram:
        """
        Construit le spectrogramme de travail:
        1) STFT
        2) normalisation par bin temps-fréquence
        """
        return self.get_spectro(input).normalize_each_bin()

    def get_spectro(self, input: MultiSignal) -> NSpectrogram:
        return input.stft(**asdict(self.stft_parameters))

    def fit_bins(
        self,
        nspectro: 'NSpectrogram',
        active_tf_mask: np.ndarray | None = None,
    ):
        """
        Exécute le clustering EM pour chaque bin de fréquence indépendamment.

        Args:
            nspectro (NSpectrogram): Le spectrogramme normalisé.
        """
        # Sxx shape: (n_micros, n_freqs, n_times)
        n_micros, n_freqs, n_times = nspectro.Sxx.shape
        self.bin_models.clear()
        self.bin_masks.clear()
        self.bin_posteriors.clear()
        self.bin_active_clusters.clear()
        self.active_frequency_mask = np.zeros(n_freqs, dtype=bool)

        print(f"Début du clustering pour {n_freqs} bins fréquentiels...")

        for f_idx in range(n_freqs):
            # 1. Extraction des données pour la fréquence f
            # X_f shape: (n_micros, n_times)
            X_f = nspectro.Sxx[:, f_idx, :]
            active_frames = (
                np.ones(n_times, dtype=bool)
                if active_tf_mask is None
                else np.asarray(active_tf_mask[f_idx], dtype=bool)
            )
            active_frames = self._keep_consecutive_active_runs(
                active_frames,
                self.em_clustering_parameters.min_active_frames_per_frequency,
            )
            min_active = max(
                self.n_sources,
                self.em_clustering_parameters.min_active_frames_per_frequency,
            )
            has_enough_active_frames = int(np.sum(active_frames)) >= min_active
            self.bin_masks[f_idx] = np.zeros((self.n_sources, n_times), dtype=int)
            self.bin_posteriors[f_idx] = np.zeros((self.n_sources, n_times), dtype=float)
            self.bin_active_clusters[f_idx] = np.zeros(self.n_sources, dtype=bool)
            if not has_enough_active_frames:
                print(f"\r frequence numéro {f_idx} ignorée", end="", flush=True)
                continue

            self.active_frequency_mask[f_idx] = True
            X_fit = X_f[:, active_frames]

            # 2. Initialisation de l'EM pour ce bin
            # On utilise les paramètres de la classe Sawada
            model = EMClustering(
                n_sources=self.n_sources,
                n_iter=self.em_clustering_parameters.n_iter,
                phi= self.em_clustering_parameters.phi,
                eps = self.em_clustering_parameters.eps
            )

            # 3. Entraînement (M-Step et E-Step alternés)
            # Cette étape estime a_i(f) et sigma_i(f)
            model.fit(X_fit)
            active_clusters = np.ones(self.n_sources, dtype=bool)

            # 4. Génération du masque binaire (Hard labeling) pour ce bin
            # mask_f shape: (n_sources, n_times)
            mask_f = np.zeros((self.n_sources, n_times), dtype=int)
            posterior_f = np.zeros((self.n_sources, n_times), dtype=float)
            posterior_active = model.posteriors
            if posterior_active is None:
                posterior_active = model.e_step(X_fit)
            posterior_f[:, active_frames] = posterior_active
            best_source = np.argmax(posterior_active, axis=0)
            for source_index in range(self.n_sources):
                if active_clusters[source_index]:
                    mask_f[source_index, active_frames] = (
                        best_source == source_index
                    )

            self.bin_models[f_idx] = model
            self.bin_masks[f_idx] = mask_f
            self.bin_posteriors[f_idx] = posterior_f
            self.bin_active_clusters[f_idx] = active_clusters
            print(f"\r frequence numéro {f_idx} terminée", end="", flush=True )
        print(" ")
        print("Clustering par bin terminé.")
    @property
    def all_centroids(self) -> np.ndarray:
        """
        Retourne tous les centroïdes estimés sous forme de tenseur.
        Shape: (n_freqs, n_micros, n_sources)
        """
        if self.nspectro_preprocessed is not None:
            n_micros, n_freqs, _ = self.nspectro_preprocessed.Sxx.shape
        elif self.bin_models:
            n_freqs = max(self.bin_models) + 1
            first_model = next(iter(self.bin_models.values()))
            n_micros = first_model.centroids.shape[0]
        else:
            return np.empty((0, 0, self.n_sources), dtype=complex)

        centroids_tensor = np.full(
            (n_freqs, n_micros, self.n_sources),
            np.nan + 1j * np.nan,
            dtype=complex,
        )
        for f_idx, model in self.bin_models.items():
            centroids_tensor[f_idx] = model.centroids

        return centroids_tensor

    def _circular_ransac_slope_bounds(self, n_micros: int) -> np.ndarray:
        slope_bound = self.em_clustering_parameters.ransac_slope_bound
        if slope_bound is None:
            slope_bound = 0.05
        slope_bound = float(abs(slope_bound))
        if not np.isfinite(slope_bound) or slope_bound <= 0.0:
            raise ValueError("ransac_slope_bound doit etre strictement positif.")

        bounds = np.tile(np.asarray([-slope_bound, slope_bound], dtype=float), (n_micros, 1))
        reference_bound = max(slope_bound * 1e-6, self.em_clustering_parameters.eps)
        bounds[0] = [-reference_bound, reference_bound]
        return bounds

    def _centroid_assignment_available_mask(self, centroids: np.ndarray) -> np.ndarray:
        n_freqs, _, n_clusters = centroids.shape
        available = np.all(np.isfinite(centroids.real) & np.isfinite(centroids.imag), axis=1)
        if self.active_frequency_mask is not None and self.active_frequency_mask.size >= n_freqs:
            available &= self.active_frequency_mask[:n_freqs, np.newaxis]

        active_clusters = self.get_final_active_clusters()
        if active_clusters.ndim == 2 and active_clusters.shape[0] >= n_freqs:
            available &= active_clusters[:n_freqs, :n_clusters]
        return available

    def align_sources_with_circular_ransac(self) -> None:
        """
        Aligne les clusters frequentiels avec le RANSAC circulaire sur les centroides.

        Les centroides complexes (F, micros, clusters) sont convertis en phases
        relatives via C_m * conj(C_1), puis le RANSAC extrait des droites de
        phase coherentes. Les labels obtenus fusionnent ensuite les masques de
        clusters en masques sources finaux.
        """
        if self.nspectro_preprocessed is None:
            raise ValueError("Aucun spectrogramme pretraite disponible.")
        if not self.bin_masks:
            raise ValueError("Aucun masque de cluster disponible.")

        centroids = self.all_centroids
        if centroids.ndim != 3 or centroids.size == 0:
            raise ValueError("Aucun centroide disponible pour le RANSAC circulaire.")

        n_freqs, n_micros, _ = centroids.shape
        frequencies = np.asarray(self.nspectro_preprocessed.f, dtype=float)[:n_freqs]
        available = self._centroid_assignment_available_mask(centroids)
        if int(np.sum(np.any(available, axis=1))) < max(2, self.n_sources):
            raise RuntimeError("Pas assez de frequences avec centroides disponibles pour le RANSAC.")
        slope_bounds = self._circular_ransac_slope_bounds(n_micros)
        print(
            "RANSAC circulaire: "
            f"{int(np.sum(np.any(available, axis=1)))}/{n_freqs} frequences utilisables, "
            f"{int(np.sum(available))} centroides disponibles, "
            f"{self.n_sources} sources, {n_micros} composantes.",
            flush=True,
        )
        print(
            "RANSAC circulaire params: "
            f"threshold={self.em_clustering_parameters.ransac_residual_threshold}, "
            f"max_trials={self.em_clustering_parameters.ransac_max_trials}, "
            f"slope_grid={self.em_clustering_parameters.ransac_slope_grid_size}, "
            f"LO={self.em_clustering_parameters.ransac_local_optimization_steps}, "
            f"slope_bound={float(np.max(np.abs(slope_bounds[:, 1]))):.4g}.",
            flush=True,
        )

        assignment = fit_centroid_source_trajectories(
            centroids,
            frequencies,
            n_sources=self.n_sources,
            slope_bounds=slope_bounds,
            residual_threshold=self.em_clustering_parameters.ransac_residual_threshold,
            component_axis=1,
            available=available,
            verbose=True,
            max_trials=self.em_clustering_parameters.ransac_max_trials,
            random_state=self.em_clustering_parameters.ransac_random_state,
            local_optimization_steps=(
                self.em_clustering_parameters.ransac_local_optimization_steps
            ),
            slope_grid_size=self.em_clustering_parameters.ransac_slope_grid_size,
            n_local_refinements=(
                self.em_clustering_parameters.ransac_n_local_refinements
            ),
            max_hypotheses_per_pair=(
                self.em_clustering_parameters.ransac_max_hypotheses_per_pair
            ),
            min_inliers=max(
                2,
                self.em_clustering_parameters.min_active_frames_per_frequency,
            ),
        )
        print(
            f"RANSAC circulaire termine: {len(assignment.models)}/{self.n_sources} trajectoires trouvees.",
            flush=True,
        )

        cluster_masks = self.get_final_masks()
        self.cluster_masks_before_assignment = cluster_masks.copy()
        source_masks = labels_to_source_masks(
            cluster_masks,
            assignment.labels,
            n_sources=self.n_sources,
            cluster_axis=0,
            aggregation="sum",
            clip=True,
        )
        cluster_posteriors = self.get_final_posteriors()
        self.cluster_posteriors_before_assignment = cluster_posteriors.copy()
        self.cluster_active_before_assignment = self.get_final_active_clusters().copy()
        source_posteriors = labels_to_source_masks(
            cluster_posteriors,
            assignment.labels,
            n_sources=self.n_sources,
            cluster_axis=0,
            aggregation="sum",
            clip=True,
        )

        for frequency_index in range(source_masks.shape[1]):
            self.bin_masks[frequency_index] = source_masks[:, frequency_index, :]
            self.bin_posteriors[frequency_index] = source_posteriors[:, frequency_index, :]
            self.bin_active_clusters[frequency_index] = np.any(
                source_masks[:, frequency_index, :] > 0.5,
                axis=1,
            )

        self.source_assignment = assignment
        self.source_assignment_labels = assignment.labels
        self.source_assignment_distances = assignment.distances
        self.source_assignment_relative_phases = assignment.relative_phases
        self.source_assignment_selected_labels = assignment.selected_labels
        self.source_assignment_slopes = np.asarray(
            [model.slope_ for model in assignment.models if model.slope_ is not None],
            dtype=float,
        )
        self.source_assignment_intercepts = np.asarray(
            [model.intercept_ for model in assignment.models if model.intercept_ is not None],
            dtype=float,
        )
        self.source_assignment_scores = np.asarray(
            [np.nan if model.score_ is None else model.score_ for model in assignment.models],
            dtype=float,
        )
        self.source_assignment_n_inliers = np.asarray(
            [model.n_inliers_ for model in assignment.models],
            dtype=int,
        )
        self.source_assignment_n_trials = np.asarray(
            [model.n_trials_ for model in assignment.models],
            dtype=int,
        )
        self.source_assignment_converged = np.asarray(
            [model.converged_ for model in assignment.models],
            dtype=bool,
        )
        self.source_assignment_frequency_inliers = np.asarray(
            [
                np.zeros(n_freqs, dtype=bool)
                if model.frequency_inliers_ is None
                else model.frequency_inliers_
                for model in assignment.models
            ],
            dtype=bool,
        )
        self.source_assignment_selected_centroids = np.asarray(
            [
                np.full(n_freqs, -1, dtype=int)
                if model.selected_centroids_ is None
                else model.selected_centroids_
                for model in assignment.models
            ],
            dtype=int,
        )

    def align_permutations(self, memory: str = 'ema'):
        """
        Résout le problème de permutation en utilisant la corrélation d'enveloppe.

        L'algorithme parcourt les fréquences et aligne les masques de chaque bin
        sur une référence accumulée (moyenne des enveloppes précédentes).

        arg: memory(str): 'ema' - l'enveloppe calculé evolue selon une ema
                          'average' - suit une moyenne arithmétique
            nspecto(Nspectrogram):  Spectrogram utilisé pour calculer l'enveloppe en utilisant les masks
                                    calculés précedemment
        """
        assert self.nspectro_preprocessed is not None, "Aucun signal processed"
        nspectro = self.nspectro_preprocessed
        n_micros, n_freqs, n_times = nspectro.Sxx.shape

        # 1. Calcul des amplitudes globales (pour le calcul des enveloppes)
        # On utilise la norme des capteurs pour avoir une enveloppe robuste
        amplitudes = np.linalg.norm(nspectro.Sxx, axis=0) # Shape: (n_freqs, n_times)

        # 2. Initialisation des enveloppes de référence (Source x Temps)
        # On initialise avec le premier bin de fréquence
        reference_envelopes = np.zeros((self.n_sources, n_times))
        for i in range(self.n_sources):
            reference_envelopes[i] = amplitudes[0, :] * self.bin_masks[0][i, :]

        # Liste des indices de permutation possibles (ex: [(0,1), (1,0)] pour 2 sources)
        perm_list = list(permutations(range(self.n_sources)))

        print("Alignement des fréquences par corrélation d'enveloppe...")

        # 3. Boucle d'alignement (on commence au bin 1)
        for f in range(1, n_freqs):
            current_masks = self.bin_masks[f] # Shape: (n_sources, n_times)

            best_perm = perm_list[0]
            max_corr = -np.inf

            # Tester toutes les permutations pour trouver celle qui maximise la corrélation
            for p in perm_list:
                total_corr = 0
                for i_source, j_cluster in enumerate(p):
                    # Enveloppe du cluster j à la fréquence f
                    env_j = amplitudes[f, :] * current_masks[j_cluster, :]

                    # Corrélation de Pearson avec la référence de la source i
                    total_corr += self._calculate_correlation(env_j, reference_envelopes[i_source])

                if total_corr > max_corr:
                    max_corr = total_corr
                    best_perm = p

            # 4. Appliquer la meilleure permutation au bin actuel
            self.bin_masks[f] = current_masks[list(best_perm), :]
            if f in self.bin_posteriors:
                self.bin_posteriors[f] = self.bin_posteriors[f][list(best_perm), :]
            if f in self.bin_active_clusters:
                self.bin_active_clusters[f] = self.bin_active_clusters[f][list(best_perm)]
            if f in self.bin_models:
                self.bin_models[f].centroids = self.bin_models[f].centroids[:, list(best_perm)]
                self.bin_models[f].variances = self.bin_models[f].variances[list(best_perm)]
                self.bin_models[f].weights = self.bin_models[f].weights[list(best_perm)]

            # 5. Mise à jour de la référence (Moyenne glissante pour stabilité)
            # Cela permet de lisser l'évolution temporelle des sources

            Source_energy_threshold = 0.05 #threshold pour savoir si une enveloppe met à jour une enveloppe doit etre prise en compte
            EMA_COEF = 0.1
            for i in range(self.n_sources):
                new_env = amplitudes[f, :] * self.bin_masks[f][i, :]
                source_energy_f = np.sum(new_env**2)

                if source_energy_f < Source_energy_threshold:
                    continue

                # On met à jour la référence (poids faible pour ne pas oublier le passé)
                if memory == 'average':
                    reference_envelopes[i] = (f * reference_envelopes[i] + new_env) / (f + 1)
                elif memory == 'ema':
                    reference_envelopes[i] = (1-EMA_COEF) * reference_envelopes[i] + EMA_COEF * new_env
                else:
                    raise ValueError("memory mode non supporté")

        print("Alignement terminé.")

    def _calculate_correlation(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calcule le coefficient de corrélation de Pearson entre deux enveloppes."""
        if np.std(v1) < 1e-6 or np.std(v2) < 1e-6:
            return 0.0
        # Soustraction de la moyenne pour centrer les enveloppes
        v1_c = v1 - np.mean(v1)
        v2_c = v2 - np.mean(v2)

        num = np.sum(v1_c * v2_c)
        den = np.sqrt(np.sum(v1_c**2) * np.sum(v2_c**2))

        return num / (den + 1e-12)

    def get_final_masks(self) -> np.ndarray:
        """Retourne le tenseur des masques alignés (Sources, Freqs, Temps)."""
        n_freqs = len(self.bin_masks)
        n_times = self.bin_masks[0].shape[1]

        final_masks = np.zeros((self.n_sources, n_freqs, n_times))
        for f in range(n_freqs):
            final_masks[:, f, :] = self.bin_masks[f]
        return final_masks

    def get_final_posteriors(self) -> np.ndarray:
        """Retourne les probabilites posterieures EM alignees (Sources, Freqs, Temps)."""
        n_freqs = len(self.bin_posteriors)
        n_times = self.bin_posteriors[0].shape[1]

        final_posteriors = np.zeros((self.n_sources, n_freqs, n_times))
        for f in range(n_freqs):
            final_posteriors[:, f, :] = self.bin_posteriors[f]
        return final_posteriors

    def get_final_active_clusters(self) -> np.ndarray:
        """Retourne les clusters Sawada conserves apres fusion locale."""
        if self.nspectro_preprocessed is not None:
            n_freqs = self.nspectro_preprocessed.Sxx.shape[1]
        elif self.bin_active_clusters:
            n_freqs = max(self.bin_active_clusters) + 1
        else:
            return np.empty((0, self.n_sources), dtype=bool)

        active_clusters = np.zeros((n_freqs, self.n_sources), dtype=bool)
        for f in range(n_freqs):
            if f in self.bin_active_clusters:
                active_clusters[f] = self.bin_active_clusters[f]
        return active_clusters

    def process_signal(self, multi_signal: MultiSignal) :
        """
        Exécute le pipeline complet : STFT -> Normalisation -> EM -> Alignement.
        Rempli les arguments         self.bin_models et self.bin_masks

        Args:
            multi_signal (MultiSignal): Le signal multicanal d'entrée.

        Returns:
            NSpectrogram: Le spectrogramme normalisé utilisé pour les calculs.
        """
        self.signal = multi_signal
        # 1. Prétraitement (STFT + Normalisation)
        raw_spectro = self.get_spectro(multi_signal)
        self.tf_energy = np.sum(np.abs(raw_spectro.Sxx) ** 2, axis=0)
        self.active_tf_mask = self._compute_active_tf_mask(self.tf_energy)
        self.nspectro_preprocessed = raw_spectro.normalize_each_bin()
        # 2. Algo : Clustering par bin (EM)
        print("Démarrage du clustering EM par bin...")
        self.fit_bins(self.nspectro_preprocessed, self.active_tf_mask)

        # 3. Algo : coherence des sources entre frequences
        alignment_method = self.em_clustering_parameters.source_alignment_method
        if alignment_method == "ransac":
            print("Alignement des sources par RANSAC circulaire...")
            self.align_sources_with_circular_ransac()
        elif alignment_method == "envelope":
            print("Alignement des permutations par enveloppes...")
            self.align_permutations()
        elif alignment_method in {"none", None}:
            print("Alignement des sources desactive.")
        else:
            raise ValueError(f"Methode d'alignement inconnue: {alignment_method}")

        return None

    def separate_source(self) -> List[MultiSignal]:
        """
        Isole les sources à partir des masques et revient dans le domaine temporel.

        Returns:
            list MultiSignal: Le signal temporel de chaque source extraite pour l'ensemble des canaux.
        """

        list_separated_sources = [] #list des sources separés

        for source_idx in range(self.n_sources):
            multi_sig_src_i = self._separate_source_i(source_idx)
            list_separated_sources.append(multi_sig_src_i)

        return list_separated_sources

    def _separate_source_i(self, source_idx) -> MultiSignal:
        """

        Args:
            source_idx (_type_): _description_

        Returns:
            MultiSignal: MultiSignal de la source i sur tout les canaux
        """
        return self.get_spectro_source_i(source_idx).istft_to_multisignal()

    def get_spectro_source_i(self, source_idx: int) -> NSpectrogram:

        if source_idx >= self.n_sources:
            raise ValueError(f"Indice {source_idx} invalide pour {self.n_sources} sources.")
        if self.signal == None:
            raise ValueError("Aucun signal n'a été process")

        Nspectro = self.get_spectro(self.signal)
        masks = self.get_final_masks() # Shape: (n_sources, n_freqs, n_times)
        mask_source = masks[source_idx][np.newaxis, :, :]
        masked_Sxx = Nspectro.Sxx * mask_source

        return NSpectrogram(
            f=Nspectro.f,
            t=Nspectro.t,
            Sxx=masked_Sxx,
            fs=Nspectro.fs,
            window=Nspectro.window,
            nperseg=Nspectro.nperseg,
            noverlap=Nspectro.noverlap,
            nfft=Nspectro.nfft,
            boundary=Nspectro.boundary,
            padded=Nspectro.padded,
            signal_lengths=Nspectro.signal_lengths
        )

    def estimate_pairwise_tdoas(
        self,
        max_lag_samples: int | None = None,
    ) -> np.ndarray:
        """
        Estime les TDOA pairwise pour chaque source separee.

        La sortie a la forme (n_sources, n_pairs). Pour quatre micros, les
        colonnes sont [M1M2, M1M3, M1M4, M2M3, M2M4, M3M4].
        Convention : M_iM_j = delay(M_j) - delay(M_i).
        """
        if self.signal is None:
            raise RuntimeError("Il faut appeler process_signal avant estimate_pairwise_tdoas.")

        separated_sources = self.separate_source()
        if not separated_sources:
            return np.empty((0, 0), dtype=float)

        n_mics = separated_sources[0].num_signals
        tdoas = np.empty(
            (len(separated_sources), _pairwise_tdoa_count(n_mics)),
            dtype=float,
        )

        for source_index, source in enumerate(separated_sources):
            source_data = source.data
            pair_index = 0
            for first in range(n_mics - 1):
                for second in range(first + 1, n_mics):
                    lag = _estimate_lag_by_correlation(
                        reference=source_data[first],
                        target=source_data[second],
                        max_lag_samples=max_lag_samples,
                    )
                    tdoas[source_index, pair_index] = lag / source.freq
                    pair_index += 1

        return tdoas

    def estimate_tdoas(
        self,
        max_lag_samples: int | None = None,
    ) -> np.ndarray:
        """Alias benchmark : renvoie les TDOA pairwise en secondes."""
        return self.estimate_pairwise_tdoas(max_lag_samples=max_lag_samples)
