from BSS.Utils.signal_class import Signal, MultiSignal, Mixture, NSpectrogram
import numpy as np 
import torch
from dataclasses import dataclass, field 
from abc import ABC, abstractmethod 
from scipy import signal as sp_signal
"""
Cahier des charges : faire un generateur de donnée à une fréquence donnée capable de generer des 
sinusoîdes, des spikes, et d'autres signaux à chercher

produit du bruit gaussien ou des spikes indépendant de chaque canal



Pour ajouter un nouveau type de signal possible, il doit herité de TypedSignal et tu dois utiliser
les fonctions register en fin de fichier 
"""


class TypedSignal(Signal, ABC):
    def __init__(self, data: np.ndarray, freq : float):
       super().__init__(data, freq) 
    signal_type : str = "generic"
    allowed_windows: tuple  = ("hann",)
    default_window : str = "hann"
    
    @classmethod
    @abstractmethod
    def generate(cls, *args): 
        raise NotImplementedError("cette fonction doit être implémentée")
    
@dataclass 
class SignalPlacement():
    signal : TypedSignal
    start_time: float
    window: str|None = "hann"
    gain : float = 1.0
    
    def __post_init__(self):
        if self.window is None : 
            self.window = self.signal.default_window 
        if self.window not in self.signal.allowed_windows: 
            raise ValueError( f"Fenetre {self.window} non autorisee pour {self.signal.signal_type}.")

    def is_compatible_with(self, other: 'SignalPlacement') -> bool:
        """
        Vérifie que deux placements portent des signaux de même fréquence.
        """
        if not isinstance(other, SignalPlacement):
            raise TypeError("other doit être une instance de SignalPlacement")
        return self.signal.freq == other.signal.freq

    def render(self)-> Signal:
        data_array = self._apply_window(self.signal)*self.gain
        latency = Signal.from_zeros(duration = self.start_time, freq = self.signal.freq)
        return Signal.concat(latency, data_array)
    
    def _apply_window(self, signal: Signal) -> Signal: 
       if self.window == None: 
           return signal.copy()
       else: 
            return signal * Signal(data = sp_signal.get_window(window=self.window, Nx = len(signal.data)), freq = signal.freq)
    
class CompositeSignal(TypedSignal): 
    signal_type : str = 'mix'

    """ Signal qui est la somme de plusieurs signaux placé à différends endroits et multiplié par différentes fenêtres"""
    def __init__(self, placements : list[SignalPlacement]):
        assert len(placements)>0, "aucun SignalPlacement ajouté"
        
        if not self.verify_frequency_coherence(placements):
            raise ValueError("Tous les SignalPlacement doivent contenir des signaux de même fréquence.")
        
        self.placements = placements
        render = self.render()
        super().__init__(data = render.data, freq = render.freq)
    
    @classmethod
    def verify_frequency_coherence(cls, placements: list[SignalPlacement]) -> bool:
        if len(placements) <= 1:
            return True
        reference = placements[0]
        return all(reference.is_compatible_with(place) for place in placements[1:])
    
    def render(self) -> Signal: 
        sig = self.placements[0].render()
        for place in self.placements[1:]:
            sig = sig + place.render()
        return sig
    
    def cut(self, start_time: float | None = None, end_time: float | None = None) -> Signal:
        """
        Retourne le signal composite découpé entre start_time et end_time.
        """
        sig = self.render()
        
        start_time = 0.0 if start_time is None else start_time
        end_time = sig.duration if end_time is None else end_time
        
        if start_time < 0 or end_time < 0:
            raise ValueError("start_time et end_time doivent être positifs.")
        if end_time < start_time:
            raise ValueError("end_time doit être supérieur ou égal à start_time.")
        
        start_idx = int(round(start_time * sig.freq))
        end_idx = int(round(end_time * sig.freq))
        return Signal(sig.data[start_idx:end_idx].copy(), sig.freq)    
        
        
    def add_placement(self, sig_placement : SignalPlacement):
        if sig_placement.is_compatible_with(self.placements[0]):
            self.placements.append(sig_placement)
            self.data = self.render()
        else :
            raise AssertionError("SignalPlacement non compatible ")
        pass  
    
    @classmethod 
    def generate(cls)-> 'CompositeSignal':
        return cls(placements = [])
    
class SinSignal(TypedSignal):
    signal_type = "sine"
    allowed_windows = ("hann",)
    default_window = "hann"
    
    def __init__(self, freq: float, data: np.ndarray, 
                 sin_freq : float, phase :float, amplitude :float): 
        super().__init__(data = data, freq = freq)
        self.sin_freq = sin_freq
        self.phase = phase
        self.amplitude = amplitude
        
    @classmethod
    def generate(cls, 
                 freq: float,
                 sin_freq : float,
                 phase: float,
                 amplitude,
                 time_duration: float) -> 'SinSignal':
        time = np.arange(0,int(round(time_duration*freq)), 1)/ freq
        data = amplitude*np.sin(2*np.pi*time*sin_freq + phase)
        return cls(freq = freq, data = data, sin_freq = sin_freq, phase = phase, amplitude = amplitude)

class SpikeSignal(TypedSignal):
    signal_type = "spike"

class GaussianNoise(TypedSignal):
    signal_type = "gaussian noise"

_SOURCE_SIGNAL_TYPE : dict[str, type['TypedSignal']]= {}
_NOISE_SIGNAL_TYPE: dict[str, type['TypedSignal']] = {}



def register_SourceSignal(sig_signal : type['TypedSignal'], ):
    if not issubclass(sig_signal, TypedSignal):
        raise TypeError("mauvais type")
    _SOURCE_SIGNAL_TYPE[sig_signal.signal_type] = sig_signal  
    
def register_NoiseSignal(sig_signal : type['TypedSignal']): 
    if not issubclass(sig_signal, TypedSignal):
        raise TypeError(...)
    _NOISE_SIGNAL_TYPE[sig_signal.signal_type] = sig_signal 
    
class SignalGenerator:
    def __init__(self): 
        return None 
    
class MixtureGenerator :
    """
    crée la mixture selon la logique que je veux avoir 
    """
    def __init__(self): 
        return 

class NoiseGenerator:
    """
    crée le bruit qu'il peut y avoir sur chacun des canaux : cela peut comprendre des bruit impulsifs
    ou du bruit gaussien
    """
    def __init__(self): 
        return 
class AudioSceneGenerator:
    """
    crée le signal recu par N microphones provenant de sources generé, d'une mixture generé 
    """
    def __init__(self): 
        return 
    def generate(self) -> 'AudioScene|None' :
        return None
    
@dataclass
class AudioSceneMetadata:
    fs:int 
    duration: float 
    n_sources :int 
    sources_types : list[str]
    n_noises : int 
    noises_types : list[str]
    max_decay : int 
    seed: int|None 
    
    def __post_init__(self):
        if self.n_sources != len(self.sources_types):
            raise ValueError("Nombre de type pas cohérent avec le nombre de sources indiqué")
        if self.n_noises != len(self.noises_types):
            raise ValueError("Nombre de type de bruit pas cohérent avec nombre de noise")

@dataclass      
class AudioScene:
    """
    Donnée pour scène acoustique :
     "mixture": MultiSignal,              # ce que les micros reçoivent
    "sources": list[Signal],             # sources propres mono
    "source_images": list[MultiSignal],  # contribution de chaque source sur chaque micro
    "mixture": Mixture,                   # filtres / retards / gains appliqués
    "metadata": {
        "fs": ...,
        "duration": ...,
        "n_sources": ...,
        "snr_db": ...,
        "delay_matrix": ...,
        "tdoa": ...,
    }
    """
    sources: MultiSignal
    mixing: Mixture
    clean_mixed: MultiSignal
    noises: MultiSignal
    mixed: MultiSignal
    metadata: AudioSceneMetadata   
