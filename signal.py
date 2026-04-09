import numpy as np 
from scipy import signal as sp_signal 
import matplotlib.pyplot as plt
class Signal:
    def __init__(self, data: np.ndarray, freq: float):
        self.data = np.array(data)
        self.freq = freq 
        
    def __add__(self, other: 'Signal') -> 'Signal' :
        if self.freq != other.freq:
            raise ValueError("Impossible d'additionner des signaux de fréquences différentes.")
        len1 = len(self.data)
        len2 = len(other.data)
        max_len = max(len1, len2)

        # On crée des copies paddées de zéros pour l'addition
        res1 = np.pad(self.data, (0, max_len - len1))
        res2 = np.pad(other.data, (0, max_len - len2))
        return Signal(res1 + res2, self.freq)
    
    def __mul__(self, other: 'Signal') -> 'Signal':
        """Surcharge de l'opérateur * (multiplication)"""
        # Cas 1 : Multiplication par un nombre (Gain / Scalaire)
        if isinstance(other, (int, float, np.number)):
            return Signal(self.data * other, self.freq)
        
        # Cas 2 : Multiplication par un autre Signal (Modulation / Enveloppe)
        elif isinstance(other, Signal):
            if self.freq != other.freq:
                raise ValueError("Les fréquences d'échantillonnage doivent être identiques.")
            
            # On aligne les longueurs (on tronque au plus court pour une modulation)
            min_len = min(len(self.data), len(other.data))
            new_data = self.data[:min_len] * other.data[:min_len]
            return Signal(new_data, self.freq)
        
        return NotImplemented
    
    @property
    def duration(self) -> float:
        return len(self.data) / self.freq
    @property
    def time(self) -> np.ndarray:
        return np.arange(0, self.duration, 1/self.freq)
    
    @classmethod
    def from_zeros(cls, duration: float, freq:float) -> 'Signal':
        """
        Args:
            duration (_type_): _description_
            freq (_type_): _description_

        Returns:
            _type_: _description_
        """
        data = np.zeros(int(duration * freq))
        return cls(data, freq)
    
    
    @classmethod
    def concat(cls, *signals : 'Signal')-> 'Signal' :
        """
        Concatène une liste de signaux.
        Usage : Signal.concat(s1, s2, s3, ...)
        """
        if not signals:
            raise ValueError("Il faut au moins un signal pour concaténer.")
        
        # On récupère la fréquence du premier pour comparer aux autres
        reference_freq = signals[0].freq
        
        # Vérification de cohérence
        for s in signals:
            if s.freq != reference_freq:
                raise ValueError(f"Fréquences incohérentes : {s.freq} != {reference_freq}")
        
        # On rassemble toutes les données (self.data) dans une liste
        all_data = [s.data for s in signals]
        
        # On crée le nouveau tableau de données concaténé
        new_data = np.concatenate(all_data)
        
        # On retourne une nouvelle instance de la classe (cls)
        return cls(new_data, reference_freq)
    @classmethod
    def convolve(cls, signal: 'Signal', impulse_response: 'list|np.ndarray|Signal', mode = "full", method = 'auto' ) -> 'Signal':
        """
        Réalise la convolution entre un signal et une réponse impulsionnelle (causal).
        
        Paramètres:
        - signal: array du signal d'origine (x)
        - impulse_response: array de la réponse impulsionnelle (h) 
        - mode 
            'full' : output of size N+L-1
            'same' : output of size N
               
        return un nouveau signal
        
        Rq : on considère ici le filtre causal ce qui implique que h[0] correspond à la réponse impultionelle en t = 0
        """
        signal_1 = signal.data
        if isinstance(impulse_response, Signal):
            if signal.freq != impulse_response.freq:
                raise ValueError("Les fréquences d'échantillonnage doivent être identiques.")
            h = impulse_response.data
        else: 
            h = np.array(impulse_response)
        
        #control de la taille de la réponse (causal)
        if mode == "full":
            resultat = Signal(sp_signal.convolve(signal_1, h, mode='full', method=method), signal.freq) 
        elif mode =="same":
            resultat = Signal(sp_signal.convolve(signal_1, h, mode='full', method = method)[:len(signal_1)], signal.freq) 

        return resultat
    
    
    @classmethod 
    def generate_multi_freq_signal(cls, duration: float, fs: float, start_time : float, end_time: float, frequencies: list, window_type: str = 'hann'):
        """
        
        Crée un signal de durée duration avec contenue fréquentiel entre deux instant

        Returns:
            Signal
        """
        t = np.arange(0, duration, 1/fs)
        signal = np.zeros_like(t)
        
        idx_start = int(start_time * fs)
        idx_end = int(end_time * fs)
        
        t_active = t[idx_start:idx_end]
        
        # Somme des composantes fréquentielles (f en Hz)
        active_signal = np.zeros_like(t_active)
        for f in frequencies:
            active_signal += np.sin(2 * np.pi * f * t_active)
        
        # Fenêtrage pour une montée/descente progressive
        window = sp_signal.get_window(window_type, len(active_signal))
        
        signal[idx_start:idx_end] = active_signal * window
        return Signal(data = signal, freq = fs )

    def fft(self, n: int|None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Calcule la transformée de Fourier discrète (DFT) du signal.
        
        Args:
            n (int, optional): Nombre de points de la FFT (zero-padding si n > len(data)).
            
        Returns:
            tuple: (fréquences en Hz, amplitudes complexes)
        """
        n = n or len(self.data)
        freqs = np.fft.fftfreq(n, 1/self.freq)
        fft_values = np.fft.fft(self.data, n)
        return freqs, fft_values

    def power_spectral_density(self, nperseg: int = 256):
        """Calcule la densité spectrale de puissance (méthode de Welch)."""
        f, psd = sp_signal.welch(self.data, self.freq, nperseg=nperseg)
        return f, psd
    
    def normalize(self, method: str = 'peak') -> 'Signal':
        """
        Normalise le signal.
        - 'peak' : ramène l'amplitude max à 1.
        - 'rms' : ramène la valeur efficace à 1.
        """
        if method == 'peak':
            new_data = self.data / np.max(np.abs(self.data))
        elif method == 'rms':
            rms = np.sqrt(np.mean(self.data**2))
            new_data = self.data / rms
        else:
            raise ValueError("Méthode inconnue")
        return Signal(new_data, self.freq)

    @property
    def energy(self) -> float:
        """Calcule l'énergie du signal : E = sum(|x[n]|^2)"""
        return np.sum(np.square(self.data)) 
    
    def stft(self, window: str = 'hann', nperseg: int = 256, noverlap: int|None = None, nfft: int|None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcule la STFT.
        
        Args:
            window (str): Type de fenêtre à utiliser (ex: 'hann', 'hamming', 'boxcar').
            nperseg (int): Longueur de chaque segment (nombre d'échantillons).
            noverlap (int, optional): Nombre de points de recouvrement entre segments. 
                                      Par défaut, nperseg // 2.
            nfft (int, optional): Longueur de la FFT utilisée, si un zero-padding est souhaité.

        Returns:
            f (np.ndarray): Tableau des fréquences d'échantillonnage.
            t (np.ndarray): Tableau des temps de segments.
            Zxx (np.ndarray): STFT du signal (valeurs complexes).
        """
        f, t, Zxx = sp_signal.stft(
            self.data, 
            fs=self.freq, 
            window=window, 
            nperseg=nperseg, 
            noverlap=noverlap, 
            nfft=nfft,
            boundary='zeros',
            padded=True
        )
        return f, t, Zxx

    def plot(self, ax=None, title="Signal", **kwargs):
        """Affiche le signal dans le domaine temporel."""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        
        ax.plot(self.time, self.data, **kwargs)
        ax.set_title(title)
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        return ax
    
    def plot_spectrogram(self, nperseg=256, ax=None, db=True):
            """
            Affiche le spectrogramme d'un signal unique.
            Rigueur : Utilise une échelle logarithmique (dB) pour la dynamique.
            """
            
            f, t, Zxx = self.stft(nperseg=nperseg)
            
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6))
            
            # Calcul de la magnitude
            magnitude = np.abs(Zxx)
            if db:
                # Ajout d'un epsilon pour éviter log(0)
                display_data = 10 * np.log10(magnitude + 1e-10)
                label = "Magnitude (dB)"
            else:
                display_data = magnitude
                label = "Magnitude (linéaire)"

            im = ax.pcolormesh(t, f, display_data, shading='gouraud', cmap='viridis')
            ax.set_title(f"Spectrogramme - fs={self.freq}Hz")
            ax.set_ylabel("Fréquence (Hz)")
            ax.set_xlabel("Temps (s)")
            
            # Gestion de la colorbar si l'axe est créé ici
            if ax.get_figure().axes:
                plt.colorbar(im, ax=ax, label=label)
                
            return im
    
class MultiSignal:
    def __init__(self, signals: list):
        """
        Conteneur pour un vecteur de signaux (X).
        
        Args:
            signals (list): Liste d'objets de type Signal.
        """
        if not signals:
            raise ValueError("La liste de signaux ne peut pas être vide.")
        
        # Validation de l'homogénéité de la fréquence d'échantillonnage
        self.freq = signals[0].freq
        if not all(s.freq == self.freq for s in signals):
            raise ValueError("Toutes les instances de Signal doivent avoir la même fréquence.")
        
        self.signals = signals
        self.num_signals = len(signals) #nombre de signal

    @property
    def data(self) -> np.ndarray:
        """
        Retourne la représentation matricielle de dimension (E, N).
        La matrice est reconstruite dynamiquement pour garantir la synchronisation
        avec les données de chaque signal et gérer le padding.
        """
        # Calcul de la longueur maximale actuelle (N)
        max_len = max(len(s.data) for s in self.signals)
        
        # Initialisation de la matrice (Nombre d'entrées E x Longueur N)
        matrix = np.zeros((self.num_signals, max_len))
        
        # Remplissage de la matrice 
        for i, s in enumerate(self.signals):
            matrix[i, :len(s.data)] = s.data
            
        return matrix

    @property
    def duration(self) -> float:
        l = max(len(s.data) for s in self.signals)
        return l / self.freq
    @property
    def time(self) -> np.ndarray:
        return np.arange(0, self.duration, 1/self.freq)
    
    def stft(self, window: str = 'hann', nperseg: int = 256, noverlap: int|None = None, nfft: int|None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcule la STFT de manière vectorisée sur la matrice de données.
        
        Args:
            window (str): Fenêtre de pondération.
            nperseg (int): Longueur de chaque segment.
            noverlap (int, optional): Nombre de points de recouvrement.
            nfft (int, optional): Longueur de la FFT.

        Returns:
            f (np.ndarray): Fréquences.
            t (np.ndarray): Temps.
            Zxx (np.ndarray): Tenseur (signaux, fréquences, temps).
        """
        # On récupère la matrice (E, N) qui gère déjà le padding
        X = self.data
        
        # scipy.signal.stft accepte des matrices en entrée.
        # Par défaut, il opère sur le dernier axe (axis=-1), ce qui correspond à nos échantillons N.
        f, t, Zxx = sp_signal.stft(
            X, 
            fs=self.freq, 
            window=window, 
            nperseg=nperseg, 
            noverlap=noverlap, 
            nfft=nfft,
            axis=-1
        )
        
        return f, t, Zxx
    
    def plot(self, overlay: bool = False, sharex: bool = True, figsize: tuple = (12, 6)):
        """
        Affiche les signaux du MultiSignal.
        
        Args:
            overlay (bool): Si True, tous les signaux sont tracés sur le même graphique.
                            Si False, chaque signal a son propre subplot.
            sharex (bool): Partage l'axe X entre les subplots (ignoré si overlay=True).
            figsize (tuple): Taille de la figure.
        """
        if overlay:
            # Cas : Tout sur le même graphique
            fig, ax = plt.subplots(figsize=figsize)
            for i, s in enumerate(self.signals):
                ax.plot(s.time, s.data, label=f"Signal {i}")
            ax.set_title("MultiSignal - Vue superposée")
            ax.set_xlabel("Temps (s)")
            ax.set_ylabel("Amplitude")
            ax.legend()
            ax.grid(True)
            return fig, ax
        else:
            # Cas : Subplots séparés 
            fig, axes = plt.subplots(self.num_signals, 1, figsize=(figsize[0], 2 * self.num_signals), sharex=sharex)
            
            if self.num_signals == 1:
                axes = [axes]
                
            for i, (s, ax) in enumerate(zip(self.signals, axes)):
                ax.plot(s.time, s.data)
                ax.set_title(f"Signal {i}")
                ax.set_ylabel("Amplitude")
                ax.grid(True)
            
            axes[-1].set_xlabel("Temps (s)")
            plt.tight_layout()
            return fig, axes
        
    def plot_spectrograms(self, nperseg=256, figsize=(12, 10), db=True):
            """
            Affiche une grille de spectrogrammes pour comparer les signaux (E).
            Rigueur : Calcul vectorisé et partage des axes pour comparaison directe.
            """
            # Utilisation du calcul vectorisé sur la matrice data (E, N)
            f, t, Zxx_multi = self.stft(nperseg=nperseg)
            
            # Création de subplots verticaux (un par signal)
            fig, axes = plt.subplots(self.num_signals, 1, figsize=figsize, sharex=True, sharey=True)
            
            # Sécurité si E=1
            if self.num_signals == 1:
                axes = [axes]
                
            for i in range(self.num_signals):
                magnitude = np.abs(Zxx_multi[i])
                if db:
                    display_data = 10 * np.log10(magnitude + 1e-10)
                else:
                    display_data = magnitude
                    
                im = axes[i].pcolormesh(t, f, display_data, shading='gouraud', cmap='magma')
                axes[i].set_title(f"Spectrogramme - Signal {i}")
                axes[i].set_ylabel("Freq (Hz)")
                
                # Une colorbar par signal pour voir les différences de gain
                fig.colorbar(im, ax=axes[i], label="Magnitude")
                
            axes[-1].set_xlabel("Temps (s)")
            plt.tight_layout()
            return fig, axes
        
    def __repr__(self):
        n_samples = self.data.shape[1]
        return f"MultiSignal(E={self.num_signals}, Fs={self.freq}Hz, Durée={n_samples/self.freq:.3f}s)"
    
class Mixture:
    def __init__(self, E: int, S: int, L: int):
        """
        Initialise un système de mélange convolutif.
        
        Args:
            E (int): Nombre de signaux d'entrée (sources).
            S (int): Nombre de signaux de sortie (capteurs).
            L (int): Taille (longueur) des filtres de réponse impulsionnelle.
        """
        self.E = E
        self.S = S
        self.L = L
        
        # Initialisation de la matrice de filtres A_ij[l]
        # Dimension (S, E, L) : 
        # - S lignes (sorties)
        # - E colonnes (entrées)
        # - L profondeur (coefficients du filtre/réponse impulsionnelle)
        self.filters = np.random.randn(S, E, L)
        
    @classmethod
    def create_delay_mixture(cls, E: int, S: int, L: int, delay_matrix: np.ndarray|None = None) -> 'Mixture':
        """
        Crée une instance de Mixture où chaque filtre est un retard pur.
        
        Args:
            E (int): Nombre de sources.
            S (int): Nombre de capteurs.
            L (int): Longueur des filtres (doit être supérieure au retard max).
            delay_matrix (np.ndarray, optional): Matrice de dimensions (S, E) contenant les indices 
                                                 des retards (entiers). Si None, les retards sont aléatoires.
                                                 
        Returns:
            Mixture: Une instance avec des filtres de type dirac.
        """
        # Initialisation de l'instance via le constructeur standard
        instance = cls(E, S, L)
        
        # Mise à zéro de tous les filtres initialisés aléatoirement par le constructeur
        instance.filters = np.zeros((S, E, L))
        
        if delay_matrix is None:
            # Génération de retards aléatoires entre 0 et L-1
            delay_matrix = np.random.randint(0, L, size=(S, E))
        else:
            # Vérification de la rigueur des dimensions
            if delay_matrix.shape != (S, E):
                raise ValueError(f"La matrice de retard doit être de taille ({S}, {E})")
            if np.max(delay_matrix) >= L:
                raise ValueError("Un retard spécifié dépasse la longueur L du filtre.")

        # Placement du dirac (valeur 1) à l'indice du retard pour chaque couple (s, e)
        for s in range(S):
            for e in range(E):
                d = int(delay_matrix[s, e])
                instance.filters[s, e, d] = 1.0
                
        return instance
    
    def apply(self, input_multi: 'MultiSignal', method: str = 'auto', mode = 'full') -> 'MultiSignal':
        """
        Applique le mélange convolutif : Y = A * X.
        Chaque sortie s est la somme des entrées e convoluées par le filtre A[s,e].
        
        Args:
            input_multi (MultiSignal): Le conteneur des signaux d'entrée.
            method (str): Méthode de convolution ('auto', 'fft', ou 'direct').
            
        Returns:
            MultiSignal: Le résultat du mélange.
        """
        # On récupère la matrice des données X (E, N)
        X = input_multi.data
        fs = input_multi.freq
        E_in, N_samples = X.shape
        
        if E_in != self.E:
            raise ValueError(f"Nombre d'entrées incohérent : reçu {E_in}, attendu {self.E}")

        # Initialisation de la matrice de sortie Y (S, N)
        Y = np.zeros((self.S, N_samples))

        for s in range(self.S):
            for e in range(self.E):
                # Extraction du filtre h_se
                h = self.filters[s, e, :]
                
                # Calcul de la convolution complète
                # L'indice 0 de h est h[0], donc la convolution est naturellement causale
                y_out = sp_signal.convolve(X[e, :], h, mode=mode, method=method)
                
                # On additionne la contribution de la source e à la sortie s
                Y[s, :] += y_out

        # Reconstruction des objets Signal pour le MultiSignal de sortie

        output_signals = [Signal(Y[i, :], fs) for i in range(self.S)]
        
        return MultiSignal(output_signals)
    
    def __repr__(self):
        return f"Mixture(Entrées={self.E}, Sorties={self.S}, Longueur du filtre={self.L})"
