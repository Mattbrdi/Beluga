import matplotlib.pyplot as plt
import numpy as np 
from numpy.typing import NDArray

from impulsive_noise_denoising.wav_reader import extract_time_slice
from impulsive_noise_denoising.stft import scipy_spectrogram, scipy_db_spectrogram, frequency_band

def _extract_canal_time_slice(canal, frame_rate, start_time, end_time):
    canal = np.asarray(canal, dtype=np.float64)
    if canal.ndim != 1:
        raise ValueError(f"Expected a 1D canal, but got an array with shape {canal.shape}")

    duration = len(canal) / frame_rate
    if start_time is None or end_time is None:
        return canal
    if start_time < 0:
        raise ValueError(f"Provided start time should be at least zero, but got {start_time}")
    if start_time >= duration:
        raise ValueError(f"Provided start time should be less than duration, but got start_time: {start_time}s and duration: {duration}s")
    if end_time <= 0:
        raise ValueError(f"Provided end time should be greater than zero, but got {end_time}")
    if end_time <= start_time:
        raise ValueError(f"Provided end time should be greater than start time, but got start_time: {start_time}s and end_time: {end_time}s")
    if end_time > duration:
        raise ValueError(f"Provided end time should be less than or equal to duration, but got end_time: {end_time}s and duration: {duration}s")

    start_idx = int(round(start_time * frame_rate))
    end_idx = int(round(end_time * frame_rate))
    return canal[start_idx:end_idx]

def plot_spectro_1D(
    canal,
    frame_rate,
    outliers=None,
    is_db=False,
    start_time=None,
    end_time=None,
    fmin=1000,
    fmax=20000,
    gain_db=0,
    range_db=80,
    n_fft=4096,
    hop_length=2048,
    cmap='magma',
):
    """Plot the spectrogram of one 1D canal."""
    extract_canal = _extract_canal_time_slice(canal, frame_rate, start_time, end_time)

    if is_db:
        freqs, times, D = scipy_db_spectrogram(extract_canal, frame_rate, n_fft, hop_length, gain_db)
        vmin = -range_db
        vmax = 0
        colorbar_format = '%+2.0f dB'
    else:
        freqs, times, D = scipy_spectrogram(extract_canal, frame_rate, n_fft, hop_length, gain_db)
        vmin = 0
        vmax = np.max(D)
        colorbar_format = None

    freqs, D = frequency_band(freqs, D, fmin, fmax)
    extent = [
        times[0],
        times[-1] if len(times) > 1 else 0,
        freqs[0],
        freqs[-1],
    ]

    fig, ax = plt.subplots(1, 1, figsize=(30, 5))
    mappable = ax.imshow(
        D,
        origin='lower',
        aspect='auto',
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if outliers is not None:
        for outlier in outliers:
            ax.vlines(outlier, freqs[0], freqs[-1], colors='red')

    ax.set_title('Spectrogramme SciPy du Canal')
    ax.set_ylim([fmin, fmax])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    plt.colorbar(mappable, ax=ax, format=colorbar_format)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

def plot_spectro(
    tetra_array,
    frame_rate,
    outliers = None,
    is_db = False,
    start_time = None,
    end_time = None,
    fmin=1000,
    fmax=20000,
    gain_db=0,
    range_db=80,
    n_fft=4096,
    hop_length=2048,
    cmap='magma',
):
    """Plot a spectrogram using NumPy for the STFT and Matplotlib for display."""
    extract_array = extract_time_slice(tetra_array, frame_rate, start_time, end_time)
    fig, axs = plt.subplots(4, 1, figsize=(30, 20), sharex=True, sharey=True)
    mappable = None

    Ds = []
    freqs_list = []
    times_list = []

    for canal in extract_array:
        if is_db:
            freqs, times, D = scipy_db_spectrogram(canal, frame_rate, n_fft, hop_length, gain_db)
        else:
            freqs, times, D = scipy_spectrogram(canal, frame_rate, n_fft, hop_length, gain_db)
      
        freqs, D = frequency_band(freqs, D, fmin, fmax)
        
        Ds.append(D)
        freqs_list.append(freqs)
        times_list.append(times)

    if is_db:
        vmin = -range_db
        vmax = 0
        colorbar_format = '%+2.0f dB'
    else:
        vmin = 0
        vmax = max(np.max(D) for D in Ds)
        colorbar_format = None

    for i, D in enumerate(Ds):
        freqs = freqs_list[i]
        times = times_list[i]
        
        extent = [
            times[0],
            times[-1] if len(times) > 1 else 0,
            freqs[0],
            freqs[-1],
        ]

        mappable = axs[i].imshow(
            D,
            origin='lower',
            aspect='auto',
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        if outliers:
            for outlier in outliers[i]:
                axs[i].vlines(outlier, freqs[0], freqs[-1], colors = 'red')

        axs[i].set_title(f'Spectrogramme SciPy du Canal {i+1}')
        axs[i].set_ylim([fmin, fmax])
        axs[i].set_ylabel('Frequency (Hz)')
        plt.colorbar(mappable, ax=axs[i], format=colorbar_format)

    axs[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_spectro_3d(
    tetra_array,
    frame_rate,
    is_db=False,
    start_time=None,
    end_time=None,
    fmin=2000,
    fmax=20000,
    gain_db=0,
    range_db=80,
    n_fft=4096,
    hop_length=2048,
    cmap='magma',
    time_step=4,
    freq_step=4,
):
    """Plot each channel spectrogram as a 3D surface."""
    extract_array = extract_time_slice(tetra_array, frame_rate, start_time, end_time)

    Ds = []
    freqs_list = []
    times_list = []

    for canal in extract_array:
        if is_db:
            freqs, times, D = scipy_db_spectrogram(canal, frame_rate, n_fft, hop_length, gain_db)
        else:
            freqs, times, D = scipy_spectrogram(canal, frame_rate, n_fft, hop_length, gain_db)

        freqs, D = frequency_band(freqs, D, fmin, fmax)

        Ds.append(D)
        freqs_list.append(freqs)
        times_list.append(times)

    if is_db:
        vmin = -range_db
        vmax = 0
        zlabel = 'Amplitude (dB)'
    else:
        vmin = 0
        vmax = max(np.max(D) for D in Ds)
        zlabel = 'Amplitude'

    fig = plt.figure(figsize=(30, 20))

    for i, D in enumerate(Ds):
        ax = fig.add_subplot(4, 1, i + 1, projection='3d')

        freqs = freqs_list[i][::freq_step]
        times = times_list[i][::time_step]
        D_plot = D[::freq_step, ::time_step]

        T, F = np.meshgrid(times, freqs)

        surface = ax.plot_surface(
            T,
            F,
            D_plot,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidth=0,
            antialiased=False,
        )

        ax.set_title(f'Spectrogramme 3D SciPy du Canal {i+1}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_zlabel(zlabel)
        ax.set_ylim([fmin, fmax])
        ax.set_zlim([vmin, vmax])
        fig.colorbar(surface, ax=ax, shrink=0.55, pad=0.05)

    plt.tight_layout()
    plt.show()
    plt.close()

def plot_spectro_per_freq(
    tetra_array,
    frame_rate,
    freq_step=1,
    min_freq_plots=None,
    max_freq_plots=None,
    is_db = False,
    start_time = None,
    end_time = None,
    fmin=2500,
    fmax=30000,
    gain_db=0,
    range_db=80,
    n_fft=4096,
    hop_length=2048,
    
    ):
    """Plot one time-series figure per frequency bin for the 4 channels."""
    extract_array = extract_time_slice(tetra_array, frame_rate, start_time, end_time)

    channel_spectrograms = []
    for canal in extract_array:
        if is_db:
            freqs, times, D = scipy_db_spectrogram(canal, frame_rate, n_fft, hop_length, gain_db)
        else:
            freqs, times, D = scipy_spectrogram(canal, frame_rate, n_fft, hop_length, gain_db)
        freqs, D = frequency_band(freqs, D, fmin, fmax)
        channel_spectrograms.append(D)

    selected_freq_indexes = np.arange(0, len(freqs), freq_step)
    if max_freq_plots is not None:
        selected_freq_indexes = selected_freq_indexes[:(max_freq_plots // freq_step)]

    if min_freq_plots is not None: 
        selected_freq_indexes = selected_freq_indexes[(min_freq_plots // freq_step):]


    ymax = np.max(D)
    for freq_idx in selected_freq_indexes:
        freq = freqs[freq_idx]
        if is_db:
            ylabel = 'Amplitude (dB)'
            ylim = [-range_db, 0]
        else:
            ylabel = 'Amplitude'
            freq_values = np.concatenate([D[freq_idx, :] for D in channel_spectrograms])
            ylim = [0, ymax]

        fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        for channel_idx, D in enumerate(channel_spectrograms):
            axs[channel_idx].plot(times, D[freq_idx, :])
            axs[channel_idx].set_title(f'Canal {channel_idx + 1} - {freq:.1f} Hz')
            axs[channel_idx].set_ylabel(ylabel)
            axs[channel_idx].set_ylim(ylim)
        axs[-1].set_xlabel('Time (s)')
        plt.show()
        plt.close()

def plot_1D_signal(x : NDArray[np.float64], sample_rate : float = None):
    x = np.array(x, dtype=np.float64)
    signal_length = len(x)
    if sample_rate:
        time = np.linspace(0, signal_length/sample_rate, signal_length)
    else:
        time = np.arange(0, signal_length)
    plt.plot(time, x)
    if sample_rate:
        plt.xlabel("time in (s)")
    else:
        plt.xlabel("time in a.u")
    plt.ylabel("amplitude in a.u")
    plt.show()

def plot_1D_signal_side_to_side(x : NDArray[np.float64], y : NDArray[np.float64], sample_rate : float = None):
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    signal_length_x = np.len(x)
    signal_length_y = np.len(y)
    if sample_rate:
        time_x = np.linspace(0, signal_length_x/sample_rate, signal_length_x)
        time_y = np.linspace(0, signal_length_y/sample_rate, signal_length_y)
    else:
        time_x = np.arange(0, signal_length_x)
        time_y = np.arange(0, signal_length_y)
    fig, (ax1, ax2) = plt.subplot(1, 2, sharey=True)
    ax1.plot(time_x, x)
    ax2.plot(time_y, y)
    if sample_rate:
        ax1.set_xlabel("time in (s)")
        ax2.set_xlabel("time in (s)")
    else:
        ax1.set_xlabel("time in a.u")
        ax2.set_xlabel("time in a.u")

    ax1.set_ylabel("amplitude in a.u")
    ax2.set_ylabel("amplitude in a.u")
    plt.show()

def plot_4D_signal(x :NDArray[np.float64], sample_rate : float = None):
    x = np.array(x, dtype=np.float64)

    if x.ndim != 2 or x.shape[0] != 4:
        raise ValueError(f"Expected x to have shape (4, N), but got {x.shape}")

    signal_length = x.shape[1]
    if sample_rate:
        time = np.linspace(0, signal_length/sample_rate, signal_length)
        xlabel = "time in (s)"
    else:
        time = np.arange(0, signal_length)
        xlabel = "time in a.u"

    fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True, sharey=True)

    for i, canal in enumerate(x):
        axs[i].plot(time, canal)
        axs[i].set_title(f"Canal {i + 1}")
        axs[i].set_ylabel("amplitude in a.u")

    axs[-1].set_xlabel(xlabel)
    plt.tight_layout()
    plt.show()
