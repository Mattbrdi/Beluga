import numpy as np 
from  numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.signal import stft
from time_frequency_mask.stft import frequency_band, scipy_db_spectrogram, scipy_spectrogram
from time_frequency_mask.config import Parameters

MAX_WAVEFORM_PLOT_POINTS = 50_000

def _downsample_for_plot(time: NDArray[np.float64], waveform: NDArray[np.float64]):
    if len(waveform) <= MAX_WAVEFORM_PLOT_POINTS:
        return time, waveform

    step = int(np.ceil(len(waveform) / MAX_WAVEFORM_PLOT_POINTS))
    return time[::step], waveform[::step]

def plot_waveform_1D(waveform : NDArray[np.float64], sampling_rate : float):
    waveform = np.array(waveform, dtype=np.float64)
    signal_length = len(waveform)
    if sampling_rate:
        time = np.linspace(0, signal_length/sampling_rate, signal_length)
    else:
        time = np.arange(0, signal_length)

    time, waveform = _downsample_for_plot(time, waveform)
    plt.plot(time, waveform)
    if sampling_rate:
        plt.xlabel("time in (s)")
    else:
        plt.xlabel("time in a.u")
    plt.ylabel("amplitude in a.u")
    plt.show()

def plot_waveform_4D(audio_array : NDArray[np.float64], sampling_rate : float):
    audio_array = np.array(audio_array, dtype=np.float64)

    if audio_array.ndim != 2:
        raise ValueError(f"Expected audio_array to have shape (M, N), but got {audio_array.shape}")

    signal_length = audio_array.shape[1]
    if sampling_rate:
        time = np.linspace(0, signal_length/sampling_rate, signal_length)
        xlabel = "time in (s)"
    else:
        time = np.arange(0, signal_length)
        xlabel = "time in a.u"

    fig, axs = plt.subplots(audio_array.shape[0], 1, figsize=(14, 10), sharex=True, sharey=False)

    axs = np.atleast_1d(axs)

    for i, canal in enumerate(audio_array):
        plot_time, plot_canal = _downsample_for_plot(time, canal)
        axs[i].plot(plot_time, plot_canal)
        axs[i].set_title(f"Canal {i + 1}")
        axs[i].set_ylabel("amplitude in a.u")

    axs[-1].set_xlabel(xlabel)
    plt.tight_layout()
    plt.show()

def plot_spectrogram_1D(waveform : NDArray[np.float64],
                        sampling_rate : float,
                        parameters : Parameters,
                        is_db = False,
                        gain_db=0,
                        range_db=80,
                    ):

    audio, stft = parameters.audio, parameters.stft

    if is_db:
        freqs, times, D = scipy_db_spectrogram(waveform, sampling_rate, stft, gain_db)
    else:
        freqs, times, D = scipy_spectrogram(waveform, sampling_rate, stft)
    freqs, D = frequency_band(freqs, D, audio.min_freq, audio.max_freq)

    if is_db:
        vmin = -range_db
        vmax = 0
        colorbar_format = '%+2.0f dB'
    else:
        
        if stft.spectrogram_type == 1:
            D = D - np.min(D)
            D = D / np.percentile(np.abs(D),99)
            D = np.clip(D, 0, 1)

        if stft.spectrogram_type == 2:
            D = D - np.min(D)
            D = D / np.percentile(np.abs(D), 95)
            D = np.clip(D, 0, 5)

        vmin = 0
        vmax = np.max(D)
        colorbar_format = None

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
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title('Spectrogramme SciPy du Canal')
    ax.set_ylim([audio.min_freq, audio.max_freq])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    plt.colorbar(mappable, ax=ax, format=colorbar_format)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

def plot_spectrogram_4D(audio_array : NDArray[np.float64],
                        sampling_rate : float,
                        parameters : Parameters,
                        mask=None,
                        is_db = False,
                        gain_db=0,
                        range_db=80,
                        ):
    audio, stft = parameters.audio, parameters.stft
    
    fig, axs = plt.subplots(parameters.array.num_mics, 1, figsize=(30, 20), sharex=True, sharey=True)
    mappable = None

    axs = np.atleast_1d(axs)

    Ds = []
    freqs_list = []
    times_list = []

    for canal in audio_array:
        if is_db:
            freqs, times, D = scipy_db_spectrogram(canal, sampling_rate, stft, gain_db)
        else:
            freqs, times, D = scipy_spectrogram(canal, sampling_rate, stft)
      
        freqs, D = frequency_band(freqs, D, audio.min_freq, audio.max_freq)
        
        if not is_db:
            if stft.spectrogram_type == 1:
                D = D - np.min(D)
                D = D / np.percentile(np.abs(D), 99)
                D = np.clip(D, 0, 1)

            if stft.spectrogram_type == 2:
                D = D - np.min(D)
                D = D / np.percentile(np.abs(D), 95)
                D = np.clip(D, 0, 5)
        
        Ds.append(D)
        freqs_list.append(freqs)
        times_list.append(times)

    if is_db:
        vmin = -range_db
        vmax = [0 for D in Ds]
        colorbar_format = '%+2.0f dB'
    else:
        vmin = 0
        vmax = [np.max(D) for D in Ds]
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
            cmap="magma",
            vmin=vmin,
            vmax=vmax[i],
        )

        if mask is not None:
            overlay = np.ma.masked_where(mask == 0, mask)            
            axs[i].imshow(overlay,
                          cmap="Blues",
                          alpha=0.45,
                          origin=mappable.origin,
                          extent=mappable.get_extent(),
                          aspect=axs[i].get_aspect()
                          )

        axs[i].set_title(f'Spectrogramme SciPy du Canal {i+1}')
        axs[i].set_ylim([audio.min_freq, audio.max_freq])
        axs[i].set_ylabel('Frequency (Hz)')
        plt.colorbar(mappable, ax=axs[i], format=colorbar_format)

    axs[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_mask(mask : NDArray[np.uint8]):
    plt.imshow(
            mask,
            origin='lower',
            aspect='auto',
            cmap="gray_r",
            vmin=0,
            vmax=1,
        )
    plt.show()
