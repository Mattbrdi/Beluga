from dataclasses import dataclass


import numpy as np
from numpy.typing import NDArray


from beamforming.geometry import Direction


@dataclass
class SearchGrid:
    theta: NDArray[np.float64]
    phi: NDArray[np.float64]

    def __post_init__(self) -> None:
        self.theta = np.atleast_1d(np.asarray(self.theta, dtype=np.float64))
        self.phi = np.atleast_1d(np.asarray(self.phi, dtype=np.float64))
        if self.theta.ndim != 1:
            raise ValueError("theta must be a 1D array.")

        if self.phi.ndim != 1:
            raise ValueError("phi must be a 1D array.")

        if len(self.theta) == 0 or len(self.phi) == 0:
            raise ValueError("SearchGrid cannot be empty.")

    @classmethod
    def full_sphere(
        cls,
        n_theta: int,
        n_phi: int,
    ) -> "SearchGrid":
        """Create angular search axes spanning the whole sphere.

        Parameters
        ----------
        n_theta :
            Number of polar-angle samples.
        n_phi : int, optional
            Number of azimuth-angle samples.

        Returns
        -------
        SearchGrid
            full-sphere angular search grid.
        """
        theta_scan = np.linspace(0, np.pi, n_theta)
        phi_scan = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        return cls(theta=theta_scan, phi=phi_scan)

    @classmethod
    def around(
        cls,
        direction: Direction,
        angular_radius: float,
        n_theta: int,
        n_phi: int,
    ) -> "SearchGrid":
        """Create a local theta/phi search grid around a direction.

        Parameters
        ----------
        direction:
            Center direction of the local search.
        angular_radius:
            Half-width of the search window in radians.
        n_theta:
            Number of polar-angle samples.
        n_phi:
            Number of azimuth-angle samples.

        Returns
        -------
        SearchGrid
            Local angular search grid.
        """
        center_theta = direction.theta
        center_phi = direction.phi

        theta_min, theta_max = max(0.0, center_theta - angular_radius), min(
            np.pi, center_theta + angular_radius
        )
        theta_scan = np.linspace(theta_min, theta_max, n_theta)

        phi_offsets = np.linspace(-angular_radius, angular_radius, n_phi)
        phi_scan = np.mod(center_phi + phi_offsets, 2 * np.pi)

        return cls(theta=theta_scan, phi=phi_scan)  # (theta, phi)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.theta), len(self.phi)

    @property
    def mesh(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return theta and phi meshgrids."""
        return np.meshgrid(self.theta, self.phi, indexing="ij")

    def direction_at(
        self,
        index: tuple[int, int],
    ) -> Direction:
        """Return the direction associated with a grid index."""
        theta_idx, phi_idx = index

        return Direction.from_spherical(
            theta=self.theta[theta_idx],
            phi=self.phi[phi_idx],
        )
