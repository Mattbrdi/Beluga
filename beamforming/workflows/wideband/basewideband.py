
from abc import abstractmethod


import numpy as np
from numpy.typing import NDArray

from beamforming.config import *

from beamforming.geometry import TetrahedralArray
from beamforming.grid import SearchGrid
from beamforming.results import PseudoSpectrumResult
from beamforming.beamformers.base import Beamformer
from beamforming.signal.stft import STFTConfig
from beamforming.workflows.workflow import BeamformingWorkflow

from beamforming.signal.stft import compute_band_stft

class WidebandBeamformer(BeamformingWorkflow):
    def __init__(self, tetrahedra : TetrahedralArray, beamformer : Beamformer, stft_config : STFTConfig):
        super().__init__(tetrahedra, beamformer)

        self.tetrahedra = tetrahedra
        self.stft_config = stft_config
        
    def compute(self, signal : NDArray[np.float64], grid : SearchGrid) -> PseudoSpectrumResult:
        freqs, _, Zxx = compute_band_stft(signal, self.stft_config, SAMPLING_RATE)
        return self.compute_from_stft(Zxx, freqs, grid)    

    @abstractmethod
    def compute_from_stft(self, stft : NDArray[np.complex128], freqs : NDArray[np.float64], grid : SearchGrid) -> PseudoSpectrumResult:
        ...