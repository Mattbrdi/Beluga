from itertools import combinations


from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


from beamforming.config import *
from beamforming.geometry import *


def _draw_tetrahedra_3D(ax: Axes3D, tetrahedra: TetrahedralArray):
    p1 = tetrahedra.p1
    p2 = tetrahedra.p2
    p3 = tetrahedra.p3
    p4 = tetrahedra.p4

    x_coords = [p1[0], p2[0], p3[0], p4[0]]
    y_coords = [p1[1], p2[1], p3[1], p4[1]]
    z_coords = [p1[2], p2[2], p3[2], p4[2]]
    for i, j in combinations(range(4), 2):
        ax.plot(
            [x_coords[i], x_coords[j]],
            [y_coords[i], y_coords[j]],
            [z_coords[i], z_coords[j]],
            color="blue",
            linewidth=2,
            label="3D Segment",
        )
    ax.scatter(x_coords, y_coords, z_coords, color="red", s=50)


def _draw_source_3D(ax: Axes3D, source: Source):
    ax.scatter(*source.position, color="yellow", s=50)


def _draw_direction_3D(ax, tetrahedra, source):
    direction_vector = source.position - tetrahedra.center
    direction_vector /= np.linalg.norm(direction_vector)

    pos = tetrahedra.center
    direction_vector *= L

    ax.quiver(
        pos[0],
        pos[1],
        pos[2],
        direction_vector[0],
        direction_vector[1],
        direction_vector[2],
        color="blue",
        arrow_length_ratio=0.15,
    )


def visualize_scene_3D(tetrahedra: TetrahedralArray, source: Source):
    """Visualize tetrahedra and source in 3D scene

    Plot the 3D coordinate of the tetrahedra and source to visualize the scene

    Parameters
    ----------
    tetrahedra : TetrahedralArray
       Geometry of the four-hydrophone array. Positions are expected
       in metres.
    source : Source
        Source waveform and three-dimensional source position. The
        position is expected in meters.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    _draw_tetrahedra_3D(ax, tetrahedra)
    _draw_source_3D(ax, source)
    _draw_direction_3D(ax, tetrahedra, source)

    plt.show()
    plt.close(fig)


def visualize_scene_2D(tetrahedra: TetrahedralArray, source: Source):
    """Visualize tetrahedra and source in 2D scene

    Plot the 3D coordinate of the tetrahedra and source to visualize the scene
    in 2D by projecting in the XY plan.

    Parameters
    ----------
    tetrahedra : TetrahedralArray
       Geometry of the four-hydrophone array. Positions are expected
       in metres.
    source : Source
        Source waveform and three-dimensional source position. The
        position is expected in meters.
    """

    p1 = tetrahedra.p1
    p2 = tetrahedra.p2
    p3 = tetrahedra.p3
    p4 = tetrahedra.p4

    x_coords = [p1[0], p2[0], p3[0], p4[0]]
    y_coords = [p1[1], p2[1], p3[1], p4[1]]

    x_source = [source.position[0]]
    y_source = [source.position[1]]
    for i, j in combinations(range(4), 2):
        plt.plot(
            [x_coords[i], x_coords[j]],
            [y_coords[i], y_coords[j]],
            color="blue",
            linewidth=2,
            label="3D Segment",
        )
    plt.scatter(x_coords, y_coords, color="red", s=50)

    plt.scatter(x_source, y_source, color="yellow", s=50)

    direction_vector = tetrahedra.center - source.position

    if np.isclose(np.linalg.norm(direction_vector), 0):
        raise ValueError(
            f"source is too close to tetrahedra center. Source position is {source.position} and tetrahedra center position is {tetrahedra.center}"
        )

    direction_vector /= np.linalg.norm(direction_vector)

    direction_vector *= 3 * L  # Normalazing with tetrahedra size

    pos = source.position

    dir = direction_vector

    plt.quiver(
        pos[0],
        pos[1],
        dir[0],
        dir[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="blue",
    )
    plt.grid()
    plt.show()
    plt.close()


def plot_results(
    power_dB: NDArray[np.float64],
    Theta: NDArray[np.float64],
    Phi: NDArray[np.float64],
    floor_dB: float = -30,
):
    """Plot power as a function theta and phi

    Parameters
    ----------
    power_dB : NDArray
        Power at each angle
    Theta : NDArray
        Polar angles array
    Phi : NDArray
        Azimuthal angles array
    floor_dB : int, optional
        Lowest plotted power, by default -30
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Restrict display range, then map [-30, 0] dB onto [0, 1].
    displayed_dB = np.clip(power_dB, floor_dB, 0)
    radius = (displayed_dB - floor_dB) / (0 - floor_dB)

    x = radius * np.sin(Theta) * np.cos(Phi)
    y = radius * np.sin(Theta) * np.sin(Phi)
    z = radius * np.cos(Theta)

    normalized_colors = (displayed_dB - floor_dB) / (0 - floor_dB)
    colors = plt.cm.viridis(normalized_colors)

    ax.plot_surface(
        x,
        y,
        z,
        facecolors=colors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=False,
    )

    colorbar_source = plt.cm.ScalarMappable(
        norm=plt.Normalize(floor_dB, 0),
        cmap="viridis",
    )
    colorbar_source.set_array([])
    fig.colorbar(colorbar_source, ax=ax, label="MVDR power (dB)")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))

    plt.show()
    plt.close(fig)


def main():
    regular_tetrahedra = TetrahedralArray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1 / 2, np.sqrt(3) / 2, 0],
            [1 / 2, np.sqrt(3) / 6, np.sqrt(2 / 3)],
        ]
    )

    source = Source(np.zeros(shape=(384000,)), position=[5, 5, 5])
    visualize_scene_2D(regular_tetrahedra, source)


if __name__ == "__main__":
    main()
