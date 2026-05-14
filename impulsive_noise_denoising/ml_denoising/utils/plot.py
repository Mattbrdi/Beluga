import matplotlib.pyplot as plt
import numpy as np 
from numpy.typing import NDArray

def plot_mask_1D(input_signal : NDArray[np.float64], frame_rate : float, impulsive_mask : NDArray[np.int8]):
    input_signal = np.asarray(input_signal)
    impulsive_mask = np.asarray(impulsive_mask)

    if np.shape(input_signal) != np.shape(impulsive_mask):
        raise ValueError(f"input_signal and impulsive_mask shapes are different"
                         f"input_signal : {np.shape(input_signal)}"
                         f"impulsive_mask : {np.shape(impulsive_mask)}"
                         )

    input_signal = np.squeeze(input_signal)
    impulsive_mask = np.squeeze(impulsive_mask)

    if input_signal.ndim != 1:
        raise ValueError(
            "plot_mask_1D expects a 1D signal or a single-channel signal. "
            f"Got shape {np.shape(input_signal)} after squeezing."
        )

    N = len(input_signal)
    duration = N / frame_rate

    times = np.linspace(0, duration, N)

    def axvspan_mask(ax, mask, **kwargs):
        mask = np.asarray(mask, dtype=bool)

        padded = np.r_[False, mask, False]
        changes = np.diff(padded.astype(int))

        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        for start, end in zip(starts, ends):
            ax.axvspan(times[start], times[end - 1], **kwargs)

    fig, ax = plt.subplots()

    ax.plot(times, input_signal)
    
    axvspan_mask(ax, impulsive_mask, color='red', alpha=0.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    plt.show()
    
