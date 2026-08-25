from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans


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
    weights: np.ndarray = field(init=False)  # Alpha_i (proportions)
    posteriors: Optional[np.ndarray] = field(init=False, default=None)

    def _initialize(self, X):
        """
        Etape d'initialisation de mes variances, poids et surout des centroids utilisés ensuite

        Args:
            X (_type_): _description_
        """
        n_micros, _ = X.shape
        # Utilisation de K-means sur les amplitudes pour une convergence plus rapide
        # ou initialisation aléatoire sur la sphère unité complexe.
        km = KMeans(n_clusters=self.n_sources, n_init=1).fit(np.abs(X.T))
        self.centroids = (
            km.cluster_centers_.T
            + 1j * np.random.randn(n_micros, self.n_sources)
        )
        # Normalisation des centroids pour qu'ils soient sur la sphère unité
        self.centroids /= np.linalg.norm(self.centroids, axis=0) + self.eps
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
            proj_sq = np.abs(self.centroids[:, i : i + 1].conj().T @ X)[0] ** 2
            dist_sq = np.maximum(0, 1 - proj_sq)

            # Terme exponentiel pondéré par la variance
            exponent = -dist_sq / (self.variances[i] + self.eps)

            # Pré-facteur de normalisation de la Gaussienne complexe sur la sphère
            # Normalisation = 1 / (pi * sigma_i^2)^(M-1)
            normalization = 1.0 / np.power(
                np.pi * (self.variances[i] + self.eps),
                n_micros - 1,
            )

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
            gamma = posteriors[i]  # Poids de chaque point pour la source i
            sum_gamma = np.sum(gamma)

            # Mise à jour du poids de mélange (alpha_i) avec prior de Dirichlet
            self.weights[i] = (
                sum_gamma + self.phi
            ) / (n_points + self.n_sources * self.phi)

            # Mise à jour du centroïde a_i (Direction de la source)
            # On cherche le vecteur propre principal de la matrice de corrélation pondérée
            weighted_X = X * np.sqrt(gamma)
            R_i = weighted_X @ weighted_X.conj().T

            # np.linalg.eigh trie les valeurs propres par ordre croissant:
            # la direction dominante est donc le dernier vecteur propre.
            _, u = np.linalg.eigh(R_i)
            self.centroids[:, i] = u[:, -1]
            self.centroids /= np.linalg.norm(self.centroids, axis=0) + self.eps

            # Mise à jour de la variance sigma_i^2
            # Moyenne pondérée des distances à la nouvelle projection
            new_proj_sq = np.abs(
                self.centroids[:, i : i + 1].conj().T @ X
            )[0] ** 2
            new_dist_sq = np.maximum(0, 1 - new_proj_sq)
            self.variances[i] = np.sum(gamma * new_dist_sq) / (
                sum_gamma * (n_micros - 1) + self.eps
            )

    def fit(self, X: np.ndarray):
        self._initialize(X)

        for _ in range(self.n_iter):
            posteriors = self.e_step(X)
            self.m_step(X, posteriors)
        self.posteriors = self.e_step(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Génère des masques binaires (Hard Labeling).

        Returns:
            np.ndarray: Masque de forme (n_sources, n_points)
        """
        _, n_points = X.shape
        scores = np.zeros((self.n_sources, n_points))

        for i in range(self.n_sources):
            # On utilise la distance à la projection comme critère de décision
            proj = self.centroids[:, i : i + 1].conj().T @ X
            scores[i] = np.abs(proj[0]) ** 2 / (self.variances[i] + self.eps)

        best_source = np.argmax(scores, axis=0)

        masks = np.zeros((self.n_sources, n_points), dtype=int)
        for i in range(self.n_sources):
            masks[i, best_source == i] = 1
        return masks
