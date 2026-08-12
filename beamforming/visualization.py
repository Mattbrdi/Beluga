import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from beamforming.results import PseudoSpectrumResult


def plot_pseudospectum(
    result: PseudoSpectrumResult,
    ax: Axes3D = None,
):
    """Plot power as a function theta and phi

    Parameters
    ----------
    result : PseudoSpectrumResult
        Result of steering algorithm on grid
    ax : axes3D, optional
        axes on which is drawn pseudospectrom, by default None
    floor_dB : float, optional
        _description_, by default -30
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    else:
        fig = ax.figure

    theta = result.grid.theta
    phi = result.grid.phi

    spectrum = result.spectrum + np.min(result.spectrum)
    scale = np.percentile(result.spectrum, 99)
    if scale != 0:
        spectrum /= scale

    displayed = np.clip(spectrum, 0.0, 5)

    radius = (displayed - 0.0) / (5 - 0.0)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    normalized_colors = (displayed - 0.0) / (5 - 0.0)
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
        norm=plt.Normalize(0, 5),
        cmap="viridis",
    )
    colorbar_source.set_array([])
    fig.colorbar(colorbar_source, ax=ax, label="Pseudospectrum (dB)")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))

    return fig, ax
