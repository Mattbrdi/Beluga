from dataclasses import dataclass


import numpy as np
from numpy.typing import NDArray


from beamforming.geometry import Direction
from beamforming.grid import SearchGrid


@dataclass
class PseudoSpectrumResult:
    grid: SearchGrid
    spectrum: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.spectrum.shape != (len(self.grid.theta), len(self.grid.phi)):
            raise ValueError(
                f"incorrect spectrum shape."
                f"Got {self.spectrum.shape} instead of {(len(self.grid.theta), len(self.grid.phi))}"
                
            )
    @property
    def doa(self) -> Direction:
        index = np.unravel_index(
            np.argmax(self.spectrum),
            self.spectrum.shape,
        )

        return self.grid.direction_at(index)
