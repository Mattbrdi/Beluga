"""Séparation convolutive par ICA fréquentielle et estimation des TDOA.

Pour chaque fréquence, les observations sont réduites à ``n_sources``,
blanchies, puis séparées par une FastICA complexe. Les permutations sont
alignées entre fréquences avant de reconstruire les images multicanales des
sources. Les TDOA sont obtenus à partir de la pente de phase de la matrice de
mélange, relativement à un microphone de référence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import permutations
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar

from ..Utils.associated_dataclasses import FrequencyIcaParameters, StftParameters
from ..Utils.signal_class import MultiSignal, NSpectrogram


def _reference_tdoas_to_pairwise(tdoas: np.ndarray) -> np.ndarray:
    """
    Convertit des TDOA relatifs a un micro de reference en TDOA pairwise.

    Entree : (n_sources, n_mics). Sortie : (n_sources, n_pairs).
    Pour quatre micros, les colonnes sont :
    [M1M2, M1M3, M1M4, M2M3, M2M4, M3M4].
    Convention : M_iM_j = delay(M_j) - delay(M_i).
    """
    values = np.asarray(tdoas)
    if values.ndim != 2:
        raise ValueError("tdoas doit avoir la forme (n_sources, n_mics).")

    n_sources, n_mics = values.shape
    pairwise = np.empty(
        (n_sources, n_mics * (n_mics - 1) // 2),
        dtype=values.dtype,
    )
    pair_index = 0
    for first in range(n_mics - 1):
        for second in range(first + 1, n_mics):
            pairwise[:, pair_index] = values[:, second] - values[:, first]
            pair_index += 1
    return pairwise


@dataclass(frozen=True)
class FrequencyICAResult:
    """Résultat complet d'un appel à :meth:`FrequencyDomainICA.process_signal`.

    ``tdoas`` est exprimé en secondes et a la forme
    ``(n_sources, n_microphones)``.
    """

    sources: list[MultiSignal]
    tdoas: np.ndarray
    reference_microphone: int

    @property
    def tdoas_samples(self) -> np.ndarray:
        """TDOA en échantillons (valeurs potentiellement fractionnaires)."""
        if not self.sources:
            return np.empty_like(self.tdoas)
        return self.tdoas * self.sources[0].freq

    @property
    def pairwise_tdoas(self) -> np.ndarray:
        """TDOA pairwise en secondes, dans l'ordre M1M2, M1M3, etc."""
        return _reference_tdoas_to_pairwise(self.tdoas)

    @property
    def pairwise_tdoas_samples(self) -> np.ndarray:
        """TDOA pairwise en echantillons."""
        if not self.sources:
            return np.empty_like(self.pairwise_tdoas)
        return self.pairwise_tdoas * self.sources[0].freq

    @property
    def pairwise_tdoa_labels(self) -> list[str]:
        """Labels des colonnes de pairwise_tdoas."""
        if self.tdoas.ndim != 2:
            return []
        n_mics = self.tdoas.shape[1]
        return [
            f"M{first + 1}M{second + 1}"
            for first in range(n_mics - 1)
            for second in range(first + 1, n_mics)
        ]


@dataclass
class FrequencyDomainICA:
    """ICA complexe indépendante par bin de fréquence.

    L'algorithme accepte un cas surdéterminé (par exemple quatre microphones
    et deux sources). ``process_signal`` exécute tout le pipeline et renvoie à
    la fois les sources séparées et les TDOA.
    """

    n_sources: int
    stft_parameters: StftParameters = field(default_factory=StftParameters)
    n_iter: int = 100
    tolerance: float = 1e-6
    max_tdoa_seconds: float = 0.01
    max_lag_samples: int | None = None
    reference_microphone: int = 0
    random_state: int | None = 0
    eps: float = 1e-10

    signal: Optional[MultiSignal] = field(default=None, init=False)
    input_spectrogram: Optional[NSpectrogram] = field(default=None, init=False)
    separated_spectrograms: list[NSpectrogram] = field(default_factory=list, init=False)
    mixing_matrices: Optional[np.ndarray] = field(default=None, init=False)
    demixing_matrices: Optional[np.ndarray] = field(default=None, init=False)
    separated_stft: Optional[np.ndarray] = field(default=None, init=False)
    bin_reliability: Optional[np.ndarray] = field(default=None, init=False)
    tdoas_: Optional[np.ndarray] = field(default=None, init=False)

    @property
    def parameters(self) -> FrequencyIcaParameters:
        return FrequencyIcaParameters(
            n_sources=self.n_sources,
            stft_parameters=self.stft_parameters,
            n_iter=self.n_iter,
            tolerance=self.tolerance,
            max_tdoa_seconds=self.max_tdoa_seconds,
            max_lag_samples=self.max_lag_samples,
            reference_microphone=self.reference_microphone,
            random_state=self.random_state,
            eps=self.eps,
        )

    def _validate_input(self, multi_signal: MultiSignal) -> None:
        n_mics = multi_signal.num_signals
        if self.n_sources < 1:
            raise ValueError("n_sources doit être strictement positif.")
        if self.n_sources > n_mics:
            raise ValueError(
                "L'ICA fréquentielle requiert au moins autant de microphones "
                "que de sources."
            )
        if not 0 <= self.reference_microphone < n_mics:
            raise ValueError("reference_microphone est hors limites.")
        if self.n_iter < 1 or self.tolerance <= 0 or self.eps <= 0:
            raise ValueError("Paramètres numériques ICA invalides.")
        if self.max_tdoa_seconds <= 0:
            raise ValueError("max_tdoa_seconds doit être strictement positif.")
        if self.max_lag_samples is not None and self.max_lag_samples < 0:
            raise ValueError("max_lag_samples doit etre positif ou nul.")

    def _tdoa_search_limit_seconds(self) -> float:
        if self.max_lag_samples is None:
            return self.max_tdoa_seconds
        if self.input_spectrogram is None:
            raise RuntimeError(
                "input_spectrogram est requis pour convertir max_lag_samples en secondes."
            )
        return self.max_lag_samples / self.input_spectrogram.fs

    @staticmethod
    def _symmetric_decorrelation(vectors: np.ndarray, eps: float) -> np.ndarray:
        """Orthonormalise les colonnes complexes de ``vectors``."""
        gram = vectors.conj().T @ vectors
        values, basis = np.linalg.eigh(gram)
        inv_sqrt = basis @ np.diag(1.0 / np.sqrt(np.maximum(values, eps))) @ basis.conj().T
        return vectors @ inv_sqrt

    def _complex_fastica(
        self, z: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Renvoie ``V`` tel que les composantes soient ``V.H @ z``."""
        n_components = z.shape[0]
        initial = rng.standard_normal((n_components, n_components))
        initial = initial + 1j * rng.standard_normal((n_components, n_components))
        vectors = self._symmetric_decorrelation(initial, self.eps)

        # Non-linéarité robuste (approximation lisse de |y|).
        alpha = 0.1
        for _ in range(self.n_iter):
            y = vectors.conj().T @ z
            radius = np.abs(y) ** 2
            g = 1.0 / np.sqrt(radius + alpha)
            g_prime = -0.5 / np.power(radius + alpha, 1.5)
            weighted = np.einsum("ct,kt->ckt", z, np.conj(y) * g)
            update = np.mean(weighted, axis=2)
            beta = np.mean(g + radius * g_prime, axis=1)
            update -= vectors * beta[np.newaxis, :]
            update = self._symmetric_decorrelation(update, self.eps)

            # Convergence indépendante de la phase et du signe des composantes.
            alignment = np.abs(np.diag(update.conj().T @ vectors))
            vectors = update
            if np.max(np.abs(1.0 - alignment)) < self.tolerance:
                break
        return vectors

    def _fit_frequency_bin(
        self, x: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Sépare une matrice ``(microphones, trames)`` à une fréquence."""
        centered = x - np.mean(x, axis=1, keepdims=True)
        covariance = centered @ centered.conj().T / max(centered.shape[1], 1)
        values, basis = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1][: self.n_sources]
        raw_selected_values = np.maximum(values[order].real, 0.0)
        selected_values = np.maximum(raw_selected_values, self.eps)
        selected_basis = basis[:, order]

        whitening = (
            np.diag(1.0 / np.sqrt(selected_values)) @ selected_basis.conj().T
        )
        z = whitening @ centered
        vectors = self._complex_fastica(z, rng)
        demixing = vectors.conj().T @ whitening
        separated = demixing @ centered
        mixing = np.linalg.pinv(demixing, rcond=self.eps)

        total_energy = float(max(np.sum(np.maximum(values.real, 0.0)), self.eps))
        retained = float(np.sum(raw_selected_values) / total_energy)
        weakest = float(raw_selected_values[-1] / max(raw_selected_values[0], self.eps))
        reliability = min(retained, 1.0) * np.sqrt(max(weakest, 0.0))
        return demixing, mixing, separated, reliability

    @staticmethod
    def _profile_correlation(first: np.ndarray, second: np.ndarray) -> float:
        a = first - np.mean(first)
        b = second - np.mean(second)
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator <= 1e-12:
            return 0.0
        return float(np.real(np.vdot(a, b)) / denominator)

    def _align_permutations(
        self, mixing: np.ndarray, separated: np.ndarray, reliability: np.ndarray
    ) -> None:
        """Aligne les sources par continuité spatiale et corrélation d'enveloppe."""
        n_freqs = mixing.shape[0]
        all_permutations = list(permutations(range(self.n_sources)))
        valid = np.flatnonzero(reliability > self.eps)
        if valid.size == 0:
            return

        # On part du bin fiable le plus bas, puis on propage vers les aigus.
        first = int(valid[0])
        previous = first
        for freq_index in range(first + 1, n_freqs):
            previous_mixing = mixing[previous]
            best_permutation = all_permutations[0]
            best_score = -np.inf
            for permutation in all_permutations:
                score = 0.0
                for source_index, candidate_index in enumerate(permutation):
                    a = previous_mixing[:, source_index]
                    b = mixing[freq_index, :, candidate_index]
                    spatial = abs(np.vdot(a, b)) / (
                        np.linalg.norm(a) * np.linalg.norm(b) + self.eps
                    )
                    envelope = self._profile_correlation(
                        np.abs(separated[source_index, previous]),
                        np.abs(separated[candidate_index, freq_index]),
                    )
                    score += 0.65 * spatial + 0.35 * envelope
                if score > best_score:
                    best_score = score
                    best_permutation = permutation
            permutation_array = np.asarray(best_permutation)
            mixing[freq_index] = mixing[freq_index][:, permutation_array]
            separated[:, freq_index] = separated[permutation_array, freq_index]
            previous = freq_index

        # Les rares bins situés avant le premier bin utile suivent ce dernier.
        for freq_index in range(first - 1, -1, -1):
            current = mixing[freq_index]
            reference = mixing[freq_index + 1]
            scores = [
                sum(
                    abs(np.vdot(reference[:, i], current[:, p[i]]))
                    / (np.linalg.norm(reference[:, i]) * np.linalg.norm(current[:, p[i]]) + self.eps)
                    for i in range(self.n_sources)
                )
                for p in all_permutations
            ]
            permutation_array = np.asarray(all_permutations[int(np.argmax(scores))])
            mixing[freq_index] = current[:, permutation_array]
            separated[:, freq_index] = separated[permutation_array, freq_index]

    def fit(self, multi_signal: MultiSignal) -> "FrequencyDomainICA":
        """Estime les matrices ICA, aligne les fréquences et calcule les TDOA."""
        self._validate_input(multi_signal)
        self.signal = multi_signal
        self.input_spectrogram = multi_signal.stft(**asdict(self.stft_parameters))
        x_stft = self.input_spectrogram.Sxx
        n_mics, n_freqs, n_frames = x_stft.shape
        rng = np.random.default_rng(self.random_state)

        demixing = np.empty((n_freqs, self.n_sources, n_mics), dtype=complex)
        mixing = np.empty((n_freqs, n_mics, self.n_sources), dtype=complex)
        separated = np.empty((self.n_sources, n_freqs, n_frames), dtype=complex)
        reliability = np.empty(n_freqs, dtype=float)

        for freq_index in range(n_freqs):
            w, a, y, score = self._fit_frequency_bin(x_stft[:, freq_index], rng)
            demixing[freq_index] = w
            mixing[freq_index] = a
            separated[:, freq_index] = y
            reliability[freq_index] = score

        self._align_permutations(mixing, separated, reliability)
        # Recalcule W après permutation pour garder les attributs cohérents.
        demixing = np.asarray([np.linalg.pinv(a, rcond=self.eps) for a in mixing])
        self.demixing_matrices = demixing
        self.mixing_matrices = mixing
        self.separated_stft = separated
        self.bin_reliability = reliability
        self.tdoas_ = self.estimate_tdoas()
        self._build_source_spectrograms()
        return self

    def _delay_from_phase(
        self, frequencies: np.ndarray, phase: np.ndarray, weights: np.ndarray
    ) -> float:
        """Régression circulaire robuste de ``phase = -2*pi*f*delay``."""
        usable = (
            np.isfinite(phase)
            & np.isfinite(weights)
            & (weights > self.eps)
            & (frequencies > 0)
        )
        if np.count_nonzero(usable) < 2:
            return 0.0
        f = frequencies[usable]
        p = phase[usable]
        w = weights[usable]
        w = w / np.sum(w)

        # Le critère circulaire évite qu'un unwrap erroné décale la pente.
        # La perte tronquée empêche quelques bins mal permutés par l'ICA de
        # tirer toute la régression vers le retard d'une autre source. Une perte
        # quadratique/cosinus non bornée est trop sensible à ce cas et donne des
        # résultats différents selon le backend BLAS utilisé.
        outlier_cap = 0.25

        def robust_loss(residual: np.ndarray) -> np.ndarray:
            return np.minimum(1.0 - np.cos(residual), outlier_cap)

        search_limit_seconds = self._tdoa_search_limit_seconds()
        grid = np.linspace(-search_limit_seconds, search_limit_seconds, 4001)
        residual = p[:, np.newaxis] + 2.0 * np.pi * f[:, np.newaxis] * grid
        costs = np.sum(w[:, np.newaxis] * robust_loss(residual), axis=0)
        best_index = int(np.argmin(costs))
        step = float(grid[1] - grid[0])
        lower = max(-search_limit_seconds, grid[best_index] - step)
        upper = min(search_limit_seconds, grid[best_index] + step)

        def objective(delay: float) -> float:
            residual_at_delay = p + 2.0 * np.pi * f * delay
            return float(np.sum(w * robust_loss(residual_at_delay)))

        if lower == upper:
            return float(grid[best_index])
        result = minimize_scalar(
            objective,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1e-12},
        )
        return float(result.x)

    def estimate_tdoas(self) -> np.ndarray:
        """Estime et renvoie ``T[source, microphone]`` en secondes."""
        if self.mixing_matrices is None or self.input_spectrogram is None:
            raise RuntimeError("Il faut appeler fit ou process_signal avant estimate_tdoas.")
        mixing = self.mixing_matrices
        frequencies = self.input_spectrogram.f
        reliability = (
            np.ones(len(frequencies))
            if self.bin_reliability is None
            else self.bin_reliability
        )
        n_mics = mixing.shape[1]
        tdoas = np.zeros((self.n_sources, n_mics), dtype=float)
        reference = self.reference_microphone

        for source_index in range(self.n_sources):
            reference_response = mixing[:, reference, source_index]
            for microphone_index in range(n_mics):
                if microphone_index == reference:
                    continue
                response = mixing[:, microphone_index, source_index]
                ratio = response / np.where(
                    np.abs(reference_response) > self.eps,
                    reference_response,
                    self.eps + 0j,
                )
                phase = np.angle(ratio)
                weights = reliability * np.sqrt(
                    np.abs(response) * np.abs(reference_response)
                )
                # Supprime les bins quasi silencieux, très instables en phase.
                cutoff = np.percentile(weights, 25) if np.any(weights > 0) else 0.0
                weights = np.where(weights >= cutoff, weights, 0.0)
                tdoas[source_index, microphone_index] = self._delay_from_phase(
                    frequencies, phase, weights
                )
        return tdoas

    def estimate_pairwise_tdoas(self) -> np.ndarray:
        """
        Renvoie les TDOA pairwise issus de l'estimation ICA par pente de phase.

        La sortie a la forme (n_sources, n_pairs). Pour quatre micros, les
        colonnes sont [M1M2, M1M3, M1M4, M2M3, M2M4, M3M4].
        """
        if self.tdoas_ is None:
            self.tdoas_ = self.estimate_tdoas()
        return _reference_tdoas_to_pairwise(self.tdoas_)

    def _build_source_spectrograms(self) -> None:
        if (
            self.input_spectrogram is None
            or self.mixing_matrices is None
            or self.separated_stft is None
        ):
            raise RuntimeError("Le modèle ICA n'est pas ajusté.")
        original = self.input_spectrogram
        self.separated_spectrograms = []
        for source_index in range(self.n_sources):
            source_images = (
                self.mixing_matrices[:, :, source_index].T[:, :, np.newaxis]
                * self.separated_stft[source_index][np.newaxis, :, :]
            )
            self.separated_spectrograms.append(
                NSpectrogram(
                    f=original.f,
                    t=original.t,
                    Sxx=source_images,
                    fs=original.fs,
                    window=original.window,
                    nperseg=original.nperseg,
                    noverlap=original.noverlap,
                    nfft=original.nfft,
                    boundary=original.boundary,
                    padded=original.padded,
                    signal_lengths=original.signal_lengths,
                )
            )

    def separate_source(self) -> list[MultiSignal]:
        """Renvoie une image multicanale temporelle par source."""
        if not self.separated_spectrograms:
            raise RuntimeError("Il faut appeler fit ou process_signal avant la séparation.")
        return [spectrogram.istft_to_multisignal() for spectrogram in self.separated_spectrograms]

    def get_spectro_source_i(self, source_index: int) -> NSpectrogram:
        if not 0 <= source_index < len(self.separated_spectrograms):
            raise ValueError("Indice de source invalide ou modèle non ajusté.")
        return self.separated_spectrograms[source_index]

    def process_signal(self, multi_signal: MultiSignal) -> FrequencyICAResult:
        """Exécute STFT, ICA, alignement, reconstruction et estimation TDOA."""
        self.fit(multi_signal)
        sources = self.separate_source()
        assert self.tdoas_ is not None
        return FrequencyICAResult(
            sources=sources,
            tdoas=self.tdoas_.copy(),
            reference_microphone=self.reference_microphone,
        )

    def separate_sources_and_tdoas(
        self, multi_signal: MultiSignal
    ) -> tuple[list[MultiSignal], np.ndarray]:
        """Raccourci pratique renvoyant directement ``(sources, tdoas)``."""
        result = self.process_signal(multi_signal)
        return result.sources, result.tdoas


# Alias plus court pour les notebooks et scripts existants.
FrequencyICA = FrequencyDomainICA


__all__ = ["FrequencyDomainICA", "FrequencyICA", "FrequencyICAResult"]
