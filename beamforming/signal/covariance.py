import numpy as np
from numpy.typing import NDArray


def spatial_covariance(
    snapshots: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Return channel covariance over the last (frames) dimension."""

    num_frames = snapshots.shape[-1]

    if num_frames == 0:
        raise ValueError("cannot compute covariance from zero frames")

    if snapshots.ndim < 2:
        raise ValueError(f"Expected (..., channels, frames), got {snapshots.shape}")

    return snapshots @ snapshots.conj().swapaxes(-1, -2) / num_frames


def diagonal_loading(R : NDArray[np.complex128], relative_loading=1e-2) -> NDArray[np.complex128]:
    loading = relative_loading * np.trace(R, axis1=-2, axis2=-1).real / R.shape[-1]
    return R + loading[:, None, None] * np.eye(R.shape[-1], dtype=R.dtype)
