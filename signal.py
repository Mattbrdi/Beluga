import numpy as np 
from scipy import signal as sp_signal 

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
    def convolve(cls, signal: 'Signal', impulse_response: 'list|np.ndarray|Signal', mode='same') -> 'Signal':
        """
        Réalise la convolution entre un signal et une réponse impulsionnelle.
        
        Paramètres:
        - signal: array du signal d'origine (x)
        - impulse_response: array de la réponse impulsionnelle (h)
        - mode: 
            'full': sortie de taille N + M - 1 (inclut les queues de convolution)
            'same': sortie de la même taille que le signal d'origine (centré)
            'valid': sortie uniquement là où les signaux se chevauchent complètement
            
        return un nouveau signal
        """
        signal_1 = signal.data
        if isinstance(impulse_response, Signal):
            if signal.freq != impulse_response.freq:
                raise ValueError("Les fréquences d'échantillonnage doivent être identiques.")
            h = impulse_response.data
        else: 
            h = np.array(impulse_response)
        resultat = Signal(np.convolve(signal_1, h, mode='same'), signal.freq)
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
        self.E = len(signals) #nombre de signal

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
        matrix = np.zeros((self.E, max_len))
        
        # Remplissage de la matrice avec padding automatique
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
    
    def __repr__(self):
        n_samples = self.data.shape[1]
        return f"MultiSignal(E={self.E}, Fs={self.freq}Hz, Durée={n_samples/self.freq:.3f}s)"
    
    
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
        self.filters = np.random.randn(S, E, L)a

    def __repr__(self):
        return f"Mixture(Entrées={self.E}, Sorties={self.S}, Longueur du filtre={self.L})"git 