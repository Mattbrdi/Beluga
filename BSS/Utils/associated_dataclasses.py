from dataclasses import dataclass, field, asdict
import numpy as np 

@dataclass
class BssParameters:
    """
    Classe de base des parametres d'un algorithme BSS.
    Heriter de cette classe permet a save_module de reconnaitre
    automatiquement le type de configuration.
    """
    pass

@dataclass
class StftParameters:
    window: str = 'hann'
    nperseg: int = 256
    noverlap: int|None = None
    nfft: int |None = None
    boundary: str | None = 'zeros'
    padded: bool = True
    
@dataclass
class EMClusteringParameters:
    n_iter: int = 20
    phi: float = 1.0  # Prior de Dirichlet pour les proportions de mélange
    eps: float = 1e-12
    energy_threshold_db_above_floor: float | None = None
    energy_floor_percentile: float = 20.0
    min_active_frames_per_frequency: int = 2
    merge_centroid_distance_scale: float | None = None
    source_alignment_method: str = "ransac"
    ransac_residual_threshold: float = 0.4
    ransac_max_trials: int = 1000
    ransac_slope_bound: float | None = 0.05
    ransac_random_state: int | None = 0
    ransac_local_optimization_steps: int = 1
    ransac_slope_grid_size: int = 60
    ransac_n_local_refinements: int = 2
    ransac_max_hypotheses_per_pair: int = 100
  
@dataclass
class SawadaBssParameters(BssParameters): 
    n_sources: int
    stft_parameters: StftParameters = field(default_factory= StftParameters)
    em_clustering_parameters : EMClusteringParameters = field(default_factory= EMClusteringParameters)
    whitening: bool = False


@dataclass
class FrequencyIcaParameters(BssParameters):
    """Paramètres de la séparation ICA dans le domaine fréquentiel."""

    n_sources: int
    stft_parameters: StftParameters = field(default_factory=StftParameters)
    n_iter: int = 100
    tolerance: float = 1e-6
    max_tdoa_seconds: float = 0.01
    max_lag_samples: int | None = None
    reference_microphone: int = 0
    random_state: int | None = 0
    eps: float = 1e-10
