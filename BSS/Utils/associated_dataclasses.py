from dataclasses import dataclass, field, asdict
import numpy as np 

@dataclass
class StftParameters:
    window: str = 'hann'
    nperseg: int = 256
    noverlap: int|None = None
    nfft: int |None = None
    boundary: str | None = None
    padded: bool = True
    
@dataclass
class EMClusteringParameters:
    n_iter: int = 20
    phi: float = 1.0  # Prior de Dirichlet pour les proportions de mélange
    eps: float = 1e-12
  
@dataclass
class SawadaBssParameters: 
    n_sources: int
    stft_parameters: StftParameters = field(default_factory= StftParameters)
    em_clustering_parameters : EMClusteringParameters = field(default_factory= EMClusteringParameters)
    whitening: bool = True 
