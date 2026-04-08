import numpy as np 

class Signal:
    def __init__(self, data: np.ndarray, freq: float):
        self.data = np.array(data)
        self.freq = freq 
        
    @property
    def duration(self):
        return len(self.data) / self.freq
    @property
    def time(self):
        return np.arange(0, self.duration, 1/self.freq)
    
    @classmethod
    def from_zeros(cls, duration: float, freq:float):
        """
        Args:
            duration (_type_): _description_
            freq (_type_): _description_

        Returns:
            _type_: _description_
        """
        data = np.zeros(int(duration * freq))
        return cls(data, freq)
    
    def __add__(self, other: 'Signal'):
        if self.freq != other.freq:
            raise ValueError("Impossible d'additionner des signaux de fréquences différentes.")
        len1 = len(self.data)
        len2 = len(other.data)
        max_len = max(len1, len2)

        # On crée des copies paddées de zéros pour l'addition
        res1 = np.pad(self.data, (0, max_len - len1))
        res2 = np.pad(other.data, (0, max_len - len2))
        return Signal(res1 + res2, self.freq)
    
    def __mul__(self, other: 'Signal'):
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
    
    @classmethod
    def concat(cls, *signals : 'Signal'):
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
    def generate_multi_freq_signal(cls, duration: float, fs: float, start_time : float, end_time: float, frequencies: list):
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
        
        # Fenêtrage de Hann pour une montée/descente progressive
        window = np.hanning(len(t_active))
        
        signal[idx_start:idx_end] = active_signal * window
        return Signal(data = signal, freq = fs )
    
    def convolve(self, impulse_response: 'list|np.ndarray|Signal', mode='same'):
        """
        Réalise la convolution entre un signal et une réponse impulsionnelle.
        
        Paramètres:
        - signal: array du signal d'origine (x)
        - impulse_response: array de la réponse impulsionnelle (h)
        - mode: 
            'full': sortie de taille N + M - 1 (inclut les queues de convolution)
            'same': sortie de la même taille que le signal d'origine (centré)
            'valid': sortie uniquement là où les signaux se chevauchent complètement
        """
        signal_1 = self.data
        if isinstance(impulse_response, Signal):
            if self.freq != impulse_response.freq:
                raise ValueError("Les fréquences d'échantillonnage doivent être identiques.")
            h = impulse_response.data
        else: 
            h = np.array(impulse_response)
        resultat = Signal(np.convolve(signal_1, h, mode='same'), self.freq)
        return resultat