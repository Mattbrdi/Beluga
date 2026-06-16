from BSS.Utils.signal_class import Signal, MultiSignal, Mixture, NSpectrogram
import numpy as np 
import torch
from dataclasses import dataclass, field 

    """
Cahier des charges : faire un generateur de donnée à une fréquence donnée capable de generer des 
sinusoîdes, des spikes, et d'autres signaux à chercher

produit du bruit gaussien ou des spikes indépendant de chaque canal



    Returns:
        _type_: _description_
    """
class SignalGenerator:
    def __init__(self): 
        return 

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
    sources_types : 
    snr_db: float
    delay_matrix : np.ndarray
    seed: int
    
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
    def __init__(self, 
                 mixed: MultiSignal, 
                 sources : list[Signal], 
                 mixing : Mixture, 
                 metadata: dict
                 ): 
        return 
