from __future__ import annotations 
import numpy as np 
from scipy import signal as sp_signal 
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
from ..Utils.signal_class import Signal, MultiSignal, NSpectrogram
from dataclasses import dataclass, field, asdict
from ..Utils.associated_dataclasses import StftParameters, EMClusteringParameters, SawadaBssParameters
from sklearn.cluster import KMeans # Utilisé uniquement pour l'initialisation rapide
from typing import List, Dict, Optional
from itertools import permutations


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
class EMClustering:
    """
    Implémentation de l'algorithme Expectation-Maximization (EM) pour le clustering 
    de sources acoustiques dans le domaine fréquentiel (Approche Sawada / LOST).
    """
    n_sources: int
    n_iter: int = 20
    phi: float = 1.0  # Prior de Dirichlet pour les proportions de mélange
    eps: float = 1e-12
    
    # Attributs calculés lors du fit
    centroids: np.ndarray = field(init=False)  # Vecteurs directeurs a_i
    variances: np.ndarray = field(init=False)  # Sigmas_i^2
    weights: np.ndarray = field(init=False)    # Alpha_i (proportions)
    posteriors: Optional[np.ndarray] = field(init=False, default=None)

    def _initialize(self, X):
        """
        Etape d'initialisation de mes variances, poids et surout des centroids utilisés ensuite

        Args:
            X (_type_): _description_
        """
        n_micros, n_points = X.shape
        # Utilisation de K-means sur les amplitudes pour une convergence plus rapide
        # ou initialisation aléatoire sur la sphère unité complexe.
        km = KMeans(n_clusters=self.n_sources, n_init=1).fit(np.abs(X.T))
        self.centroids = km.cluster_centers_.T + 1j * np.random.randn(n_micros, self.n_sources)
        # Normalisation des centroids pour qu'ils soient sur la sphère unité
        self.centroids /= (np.linalg.norm(self.centroids, axis=0) + self.eps)
        self.variances = np.ones(self.n_sources)
        self.weights = np.ones(self.n_sources) / self.n_sources
        
        
    def e_step(self, X: np.ndarray) -> np.ndarray:
        """
        Étape d'Espérance : Calcule les probabilités postérieures P(Ci | x, theta).
        
        Args:
            X (np.ndarray): Spectrogramme normalisé (n_micros, n_points).
            
        Returns:
            np.ndarray: Matrice des posteriors (n_sources, n_points).
        """
        n_micros, n_points = X.shape
        posteriors = np.zeros((self.n_sources, n_points))

        for i in range(self.n_sources):
            # Distance à la projection : 1 - |a_i^H * x|^2
            proj_sq = np.abs(self.centroids[:, i:i+1].conj().T @ X)[0]**2
            dist_sq = np.maximum(0, 1 - proj_sq)
            
            # Terme exponentiel pondéré par la variance
            exponent = -dist_sq / (self.variances[i] + self.eps)
            
            # Pré-facteur de normalisation de la Gaussienne complexe sur la sphère
            # Normalisation = 1 / (pi * sigma_i^2)^(M-1)
            normalization = 1.0 / (np.power(np.pi * (self.variances[i] + self.eps), n_micros - 1))
            
            # Calcul de la probabilité non-normalisée (Prior * Vraisemblance)
            posteriors[i] = self.weights[i] * normalization * np.exp(exponent)

        # Normalisation finale pour que chaque colonne somme à 1
        return posteriors / (np.sum(posteriors, axis=0) + self.eps)
    
    def m_step(self, X: np.ndarray, posteriors: np.ndarray):
        """
        Étape de Maximisation : Met à jour les centroïdes, variances et poids.
        
        Args:
            X (np.ndarray): Spectrogramme normalisé (n_micros, n_points).
            posteriors (np.ndarray): Matrice issue de l'e_step (n_sources, n_points).
        """
        n_micros, n_points = X.shape

        for i in range(self.n_sources):
            gamma = posteriors[i] # Poids de chaque point pour la source i
            sum_gamma = np.sum(gamma)
            
            #  Mise à jour du poids de mélange (alpha_i) avec prior de Dirichlet
            self.weights[i] = (sum_gamma + self.phi) / (n_points + self.n_sources * self.phi)
            
            #  Mise à jour du centroïde a_i (Direction de la source)
            # On cherche le vecteur propre principal de la matrice de corrélation pondérée
            weighted_X = X * np.sqrt(gamma)
            R_i = weighted_X @ weighted_X.conj().T
            
            # np.linalg.eigh trie les valeurs propres par ordre croissant:
            # la direction dominante est donc le dernier vecteur propre.
            _, u = np.linalg.eigh(R_i)
            self.centroids[:, i] = u[:, -1]
            self.centroids /= (np.linalg.norm(self.centroids, axis=0) + self.eps)
            
            #  Mise à jour de la variance sigma_i^2
            # Moyenne pondérée des distances à la nouvelle projection
            new_proj_sq = np.abs(self.centroids[:, i:i+1].conj().T @ X)[0]**2
            new_dist_sq = np.maximum(0, 1 - new_proj_sq)
            self.variances[i] = np.sum(gamma * new_dist_sq) / (sum_gamma * (n_micros - 1) + self.eps)
            
        
    def fit(self, X: np.ndarray):
        self._initialize(X) # Initialisation (K-means ou aléatoire)
        
        for _ in range(self.n_iter):
            # Cycle EM
            posteriors = self.e_step(X)
            self.m_step(X, posteriors)
        self.posteriors = self.e_step(X)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Génère des masques binaires (Hard Labeling).
        
        Returns:
            np.ndarray: Masque de forme (n_sources, n_points)
        """
        n_micros, n_points = X.shape
        scores = np.zeros((self.n_sources, n_points))
        
        for i in range(self.n_sources):
            # On utilise la distance à la projection comme critère de décision
            proj = self.centroids[:, i:i+1].conj().T @ X
            scores[i] = np.abs(proj[0])**2 / (self.variances[i] + self.eps)
            
        # Winner-Takes-All
        best_source = np.argmax(scores, axis=0)
        
        masks = np.zeros((self.n_sources, n_points), dtype=int)
        for i in range(self.n_sources):
            masks[i, best_source == i] = 1
        return masks


@dataclass
class SawadaBSS:
        
    """
    Algorithme de Sawada: 
    
    préprocess:
    - Apllique la stft pour obtenir le NSpectrogram
    - Normalise par bin le Nspectrogram
    - Blanchiment 
    
    algo:
    - clustering
    - alignement des cluster entre les fréquence
    
    - utilisation des mask obtenue pour obtenir le Nspectrogram pour une source donnée
    - retour dans le domaine temporel

    """ 
    n_sources: int
    stft_parameters: StftParameters = field(default_factory= StftParameters)
    em_clustering_parameters : EMClusteringParameters = field(default_factory= EMClusteringParameters)
    
    # Paramètres avec valeurs par défaut
    whitening: bool = True 
    
    # État de l'algorithme (Champs calculés plus tard)
    # On utilise default_factory pour les dictionnaires vides
    signal: Optional['MultiSignal'] = None
    nspectro_normalized_unwhitened: Optional[NSpectrogram] = None
    nspectro_preprocessed: Optional[NSpectrogram] = None
    bin_models: Dict[int, 'EMClustering'] = field(default_factory=dict)
    bin_masks: Dict[int, np.ndarray] = field(default_factory=dict)
    bin_posteriors: Dict[int, np.ndarray] = field(default_factory=dict)
    bin_active_clusters: Dict[int, np.ndarray] = field(default_factory=dict)
    tf_energy: Optional[np.ndarray] = None
    active_tf_mask: Optional[np.ndarray] = None
    energy_threshold_db: Optional[float] = None
    
    eigenvalues_matrix: Optional[np.ndarray] = None
    eigenvector_matrix: Optional[np.ndarray] = None
    
    @property
    def parameters(self) -> SawadaBssParameters:
        return SawadaBssParameters(
            n_sources=self.n_sources,
            stft_parameters=self.stft_parameters,
            em_clustering_parameters=self.em_clustering_parameters,
            whitening=self.whitening,
        )

    def _compute_active_tf_mask(self, tf_energy: np.ndarray) -> np.ndarray:
        threshold_margin = self.em_clustering_parameters.energy_threshold_db_above_floor
        active_mask = np.ones_like(tf_energy, dtype=bool)
        self.energy_threshold_db = None
        if threshold_margin is None:
            return active_mask

        energy_db = 10 * np.log10(tf_energy + self.em_clustering_parameters.eps)
        floor_db = np.percentile(
            energy_db,
            self.em_clustering_parameters.energy_floor_percentile,
        )
        self.energy_threshold_db = float(floor_db + threshold_margin)
        return energy_db >= self.energy_threshold_db
    
    
    
    def preprocess(self, input: MultiSignal) -> NSpectrogram:
        """
        Construit le spectrogramme de travail:
        1) STFT
        2) normalisation par bin temps-fréquence
        3) blanchiment optionnel
        """
        spectro = self.get_spectro(input).normalize_each_bin()
        if self.whitening:
            spectro = self.apply_whitening(spectro)
        return spectro
    
    def get_spectro(self, input: MultiSignal) -> NSpectrogram:
        return input.stft(**asdict(self.stft_parameters))
    
    def apply_whitening(self, spectro: NSpectrogram) -> NSpectrogram:
        n_micros, n_freqs, n_times = spectro.Sxx.shape
        whitened_sxx = np.zeros_like(spectro.Sxx)
        self.eigenvalues_matrix = np.zeros((n_freqs, n_micros), dtype=float)
        self.eigenvector_matrix = np.zeros((n_freqs, n_micros, n_micros), dtype=complex)

        for frequency_index in range(n_freqs):
            X_f = spectro.Sxx[:, frequency_index, :]
            spatial_correlation = (X_f @ X_f.conj().T) / max(n_times, 1)
            eigenvalues, eigenvectors = np.linalg.eigh(spatial_correlation)
            self.eigenvalues_matrix[frequency_index] = eigenvalues
            self.eigenvector_matrix[frequency_index] = eigenvectors

            inv_sqrt = 1.0 / np.sqrt(
                np.maximum(eigenvalues, self.em_clustering_parameters.eps)
            )
            whitening_matrix = np.diag(inv_sqrt) @ eigenvectors.conj().T
            whitened_sxx[:, frequency_index, :] = whitening_matrix @ X_f

        return NSpectrogram(
            f=spectro.f,
            t=spectro.t,
            Sxx=whitened_sxx,
            fs=spectro.fs,
            window=spectro.window,
            nperseg=spectro.nperseg,
            noverlap=spectro.noverlap,
            nfft=spectro.nfft,
            boundary=spectro.boundary,
            padded=spectro.padded,
            signal_lengths=spectro.signal_lengths,
        ).normalize_each_bin()

    def _merge_close_clusters(self, model: EMClustering, X: np.ndarray) -> np.ndarray:
        """
        Fusionne les clusters dont les directions sont indiscernables localement.

        Deux clusters sont fusionnes si leur distance directionnelle
        1 - |a_i^H a_j|^2 est plus petite que la dispersion moyenne des clusters,
        multipliee par merge_centroid_distance_scale.
        """
        merge_scale = self.em_clustering_parameters.merge_centroid_distance_scale
        active_clusters = np.ones(self.n_sources, dtype=bool)
        if merge_scale is None or self.n_sources < 2 or model.posteriors is None:
            return active_clusters

        centroids = model.centroids / (
            np.linalg.norm(model.centroids, axis=0, keepdims=True) + model.eps
        )
        directional_distances = 1.0 - np.abs(centroids.conj().T @ centroids) ** 2

        parent = np.arange(self.n_sources)

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return int(index)

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for left in range(self.n_sources):
            for right in range(left + 1, self.n_sources):
                dispersion = 0.5 * (model.variances[left] + model.variances[right])
                threshold = float(merge_scale) * max(float(dispersion), model.eps)
                distance = float(np.real(directional_distances[left, right]))
                if np.isfinite(distance) and distance <= threshold:
                    union(left, right)

        groups: dict[int, list[int]] = {}
        for source_index in range(self.n_sources):
            groups.setdefault(find(source_index), []).append(source_index)

        posteriors = model.posteriors.copy()
        for group in groups.values():
            if len(group) <= 1:
                continue

            group_array = np.asarray(group, dtype=int)
            posterior_mass = np.sum(posteriors[group_array], axis=1)
            if np.sum(posterior_mass) <= model.eps:
                keeper = int(group_array[np.argmax(model.weights[group_array])])
            else:
                keeper = int(group_array[np.argmax(posterior_mass)])

            merged_gamma = np.sum(posteriors[group_array], axis=0)
            merged_weight = float(np.sum(model.weights[group_array]))

            for member in group_array:
                if member == keeper:
                    continue
                posteriors[member] = 0.0
                model.weights[member] = 0.0
                model.variances[member] = np.inf
                model.centroids[:, member] = model.centroids[:, keeper]
                active_clusters[member] = False

            posteriors[keeper] = merged_gamma
            model.weights[keeper] = merged_weight

            sum_gamma = float(np.sum(merged_gamma))
            if sum_gamma <= model.eps:
                continue

            weighted_X = X * np.sqrt(merged_gamma)
            R_i = weighted_X @ weighted_X.conj().T
            _, eigenvectors = np.linalg.eigh(R_i)
            model.centroids[:, keeper] = eigenvectors[:, -1]
            model.centroids[:, keeper] /= (
                np.linalg.norm(model.centroids[:, keeper]) + model.eps
            )

            new_proj_sq = np.abs(model.centroids[:, keeper:keeper + 1].conj().T @ X)[0] ** 2
            new_dist_sq = np.maximum(0, 1 - new_proj_sq)
            model.variances[keeper] = np.sum(merged_gamma * new_dist_sq) / (
                sum_gamma * (X.shape[0] - 1) + model.eps
            )

        model.posteriors = posteriors / (np.sum(posteriors, axis=0, keepdims=True) + model.eps)
        return active_clusters
    
    
    def fit_bins(
        self,
        nspectro: 'NSpectrogram',
        active_tf_mask: np.ndarray | None = None,
    ):
        """
        Exécute le clustering EM pour chaque bin de fréquence indépendamment.
        
        Args:
            nspectro (NSpectrogram): Le spectrogramme normalisé et blanchi.
        """
        # Sxx shape: (n_micros, n_freqs, n_times)
        n_micros, n_freqs, n_times = nspectro.Sxx.shape
        
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
            min_active = max(
                self.n_sources,
                self.em_clustering_parameters.min_active_frames_per_frequency,
            )
            has_enough_active_frames = int(np.sum(active_frames)) >= min_active
            X_fit = X_f[:, active_frames] if has_enough_active_frames else X_f
            
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
            if has_enough_active_frames:
                active_clusters = self._merge_close_clusters(model, X_fit)
            
            # 4. Génération du masque binaire (Hard labeling) pour ce bin
            # mask_f shape: (n_sources, n_times)
            mask_f = np.zeros((self.n_sources, n_times), dtype=int)
            posterior_f = np.zeros((self.n_sources, n_times), dtype=float)
            if has_enough_active_frames:
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
        n_freqs = len(self.bin_models)
        n_micros = self.bin_models[0].centroids.shape[0]
        
        centroids_tensor = np.zeros((n_freqs, n_micros, self.n_sources), dtype=complex)
        for f_idx, model in self.bin_models.items():
            centroids_tensor[f_idx] = model.centroids
            
        return centroids_tensor
    
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
        n_freqs = len(self.bin_models)
        active_clusters = np.ones((n_freqs, self.n_sources), dtype=bool)
        for f in range(n_freqs):
            if f in self.bin_active_clusters:
                active_clusters[f] = self.bin_active_clusters[f]
        return active_clusters
        
    def process_signal(self, multi_signal: MultiSignal) :
        """
        Exécute le pipeline complet : STFT -> Normalisation -> Blanchiment -> EM -> Alignement.
        Rempli les arguments         self.bin_models et self.bin_masks
        
        Args:
            multi_signal (MultiSignal): Le signal multicanal d'entrée.
            
        Returns:
            NSpectrogram: Le spectrogramme blanchi utilisé pour les calculs.
        """
        self.signal = multi_signal
        # 1. Prétraitement (STFT + Normalisation + Blanchiment)
        raw_spectro = self.get_spectro(multi_signal)
        self.tf_energy = np.sum(np.abs(raw_spectro.Sxx) ** 2, axis=0)
        self.active_tf_mask = self._compute_active_tf_mask(self.tf_energy)
        nspectro_normalized = raw_spectro.normalize_each_bin()
        self.nspectro_normalized_unwhitened = nspectro_normalized
        nspectro_preprocessed = nspectro_normalized
        if self.whitening:
            nspectro_preprocessed = self.apply_whitening(nspectro_preprocessed)
        self.nspectro_preprocessed = nspectro_preprocessed
        # 2. Algo : Clustering par bin (EM)
        print("Démarrage du clustering EM par bin...")
        self.fit_bins(nspectro_preprocessed, self.active_tf_mask)
        
        # 3. Algo : Alignement des permutations entre les fréquences
        print("Alignement des permutations...")
        self.align_permutations()
        
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
