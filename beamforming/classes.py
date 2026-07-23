import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
from beamforming.configuration import *

Vector3 = NDArray[np.float64]


def as_vector3(value) -> Vector3:
    vector = np.asarray(value, dtype=np.float64)

    if vector.shape != (3,):
        raise ValueError(f"Expected shape (3,), got {vector.shape}")

    return vector


@dataclass
class Tetrahedra:
    positions: NDArray[np.float64]

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float64)

        if self.positions.shape != (4, 3):
            raise ValueError(
                f"positions must have shape (4,3), got {self.positions.shape}"
            )

        L12 = np.linalg.norm(self.positions[0] - self.positions[1])
        L13 = np.linalg.norm(self.positions[0] - self.positions[2])
        L14 = np.linalg.norm(self.positions[0] - self.positions[3])
        L23 = np.linalg.norm(self.positions[1] - self.positions[2])
        L24 = np.linalg.norm(self.positions[1] - self.positions[3])
        L34 = np.linalg.norm(self.positions[2] - self.positions[3])

        if (
            not np.isclose(L12, L13)
            or not np.isclose(L13, L14)
            or not np.isclose(L14, L23)
            or not np.isclose(L23, L24)
            or not np.isclose(L24, L34)
        ):
            raise ValueError("Tetrahedra is not reguliar got incorrect positions")

    # @classmethod
    # def from_length_base_centroid(cls, radius : float, height) -> Tetrahedra:
    #     p1 = np.asarray([])
    #     p2 = np.asarray([])
    #     p3 = np.asarray([])
    #     p4 = np.asarray([])
    #     return cls()

    @classmethod
    def from_length_tetrahedra_centroid(cls, length: float):

        if length <= 0:
            raise ValueError(
                f"Incorrect value for length smaller or equal than 0, got {length}"
            )

        a = length / (2 * np.sqrt(2))
        p1 = np.asarray([1, 1, 1]) * a
        p2 = np.asarray([1, -1, -1]) * a
        p3 = np.asarray([-1, 1, -1]) * a
        p4 = np.asarray([-1, -1, 1]) * a
        return cls([p1, p2, p3, p4])

    @property
    def p1(self) -> Vector3:
        return np.asarray(self.positions[0])

    @property
    def p2(self) -> Vector3:
        return np.asarray(self.positions[1])

    @property
    def p3(self) -> Vector3:
        return np.asarray(self.positions[2])

    @property
    def p4(self) -> Vector3:
        return np.asarray(self.positions[3])

    @property
    def center(self) -> Vector3:
        return (self.p1 + self.p2 + self.p3 + self.p4) / 4

    def rotate(self, rotation_matrix: NDArray[np.float64]):
        if not rotation_matrix.shape == (3, 3):
            raise ValueError(
                f"Provided matrix is not of shape (3,3), got {rotation_matrix.shape} instead."
            )

        if not np.isclose(
            rotation_matrix @ rotation_matrix.transpose(),
            np.diag([1, 1, 1]).astype(np.float64),
        ):
            raise ValueError(f"Provided matrix is not a rotation.")
        p1, p2, p3, p4 = self.p1, self.p2, self.p3, self.p4
        center = (p1 + p2 + p3 + p4) / 4

        p1 -= center
        p2 -= center
        p3 -= center
        p4 -= center

        p1 = np.matmul(rotation_matrix, p1)
        p2 = np.matmul(rotation_matrix, p2)
        p3 = np.matmul(rotation_matrix, p3)
        p4 = np.matmul(rotation_matrix, p4)

        p1 += center
        p2 += center
        p3 += center
        p4 += center

        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4


class AudioArray:
    pass


@dataclass
class Source:
    signal: NDArray[np.float64]
    position: NDArray[np.float64]
    sampling_rate: float = SAMPLING_RATE

    def __post_init__(self) -> None:
        self.signal = np.asarray(self.signal, dtype=np.float64)
        self.position = np.asarray(self.position, dtype=np.float64)

        if self.position.shape != (3,):
            raise ValueError(
                f"position must have shape (3,), got {self.positions.shape}"
            )

        if self.signal.ndim != 1:
            raise ValueError(
                f"signal must be 1 dimensionnal array, got {self.signal.ndim}"
            )
