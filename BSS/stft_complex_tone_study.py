from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .associated_dataclasses import StftParameters
from .signal_class import NSpectrogram, Signal
"""
Ce script sert à justifier l'utilisation de SAWADA, notamment le fait qu'un clustering par bande de fréquence est possible 
cad que l'angle de la stft ne varie pas trop dans le temps et que l'angle general du vecteur contenant les stft des différends
signaux translatée ne varie pas non plus (rapport des modules et différence des arguments + arguments relativement constant dans le temps).

Pour le moment, chaque composante est stable dans le temps en terme de direction dans C (on peut le montrer par le calcul également)
Pour ce qui est du module, il n'y a pas de raisons qu'elles evoluent différement

"""

@dataclass
class PlotZoom:
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None


@dataclass
class StudyPlotZoom:
    time_signal_s1: PlotZoom = field(default_factory=PlotZoom)
    time_signal_s2: PlotZoom = field(default_factory=PlotZoom)
    phase_vs_time_s1: PlotZoom = field(default_factory=PlotZoom)
    phase_vs_frequency_s1: PlotZoom = field(default_factory=PlotZoom)
    magnitude_vs_time_s1: PlotZoom = field(default_factory=PlotZoom)
    magnitude_vs_frequency_s1: PlotZoom = field(default_factory=PlotZoom)
    phase_vs_time_s2: PlotZoom = field(default_factory=PlotZoom)
    phase_vs_frequency_s2: PlotZoom = field(default_factory=PlotZoom)
    magnitude_vs_time_s2: PlotZoom = field(default_factory=PlotZoom)
    magnitude_vs_frequency_s2: PlotZoom = field(default_factory=PlotZoom)
    phase_difference_vs_time: PlotZoom = field(default_factory=PlotZoom)
    magnitude_ratio_vs_time: PlotZoom = field(default_factory=PlotZoom)


@dataclass
class ComplexToneStftStudyConfig:
    fs: float = 8000.0
    duration: float = 1.0
    signal_window_duration: float = 0.5
    signal_window_center: float | None = None
    stft_parameters: StftParameters = field(
        default_factory=lambda: StftParameters(window="hann", nperseg=256, noverlap=192, nfft=512)
    )
    target_positive_bin: int = 24
    n_neighbor_bins: int = 2
    n_time_slices: int = 5
    unwrap_phase: bool = True
    time_shift_samples: int = 16


@dataclass
class ComplexToneStftStudyResult:
    config: ComplexToneStftStudyConfig
    signal_s1: Signal
    signal_s2: Signal
    spectrogram_s1: NSpectrogram
    spectrogram_s2: NSpectrogram
    fc: float
    target_bin_index: int
    signal_window_start: float
    signal_window_end: float
    time_shift_seconds: float
    selected_freq_indices: np.ndarray
    selected_time_indices_s1: np.ndarray
    selected_time_indices_s2: np.ndarray


def _get_stft_kwargs(stft_parameters: StftParameters) -> dict:
    return {
        "window": stft_parameters.window,
        "nperseg": stft_parameters.nperseg,
        "noverlap": stft_parameters.noverlap,
        "nfft": stft_parameters.nfft,
    }


def _compute_reference_frequency_grid(config: ComplexToneStftStudyConfig) -> np.ndarray:
    dummy_signal = Signal(np.zeros(int(config.duration * config.fs), dtype=complex), config.fs)
    spectrogram = dummy_signal.stft(**_get_stft_kwargs(config.stft_parameters))
    return spectrogram.f


def _get_positive_frequency_indices(frequencies: np.ndarray) -> np.ndarray:
    positive_indices = np.where(frequencies >= 0)[0]
    if positive_indices.size == 0:
        raise ValueError("La grille fréquentielle STFT ne contient aucune fréquence positive.")
    return positive_indices


def _select_target_bin_index(
    frequencies: np.ndarray,
    target_positive_bin: int,
) -> int:
    positive_indices = _get_positive_frequency_indices(frequencies)
    if target_positive_bin < 0 or target_positive_bin >= positive_indices.size:
        raise ValueError(
            f"target_positive_bin={target_positive_bin} hors bornes pour "
            f"{positive_indices.size} bins positifs disponibles."
        )
    return int(positive_indices[target_positive_bin])


def generate_complex_tone(
    config: ComplexToneStftStudyConfig,
) -> tuple[Signal, float, int, float, float]:
    frequencies = _compute_reference_frequency_grid(config)
    target_bin_index = _select_target_bin_index(frequencies, config.target_positive_bin)
    fc = float(frequencies[target_bin_index])

    n_samples = int(config.duration * config.fs)
    time = np.arange(n_samples) / config.fs
    complex_tone = np.exp(1j * 2 * np.pi * fc * time)
    window_duration = min(config.signal_window_duration, config.duration)
    if window_duration <= 0:
        raise ValueError("signal_window_duration doit etre strictement positive.")

    window_length = max(1, int(round(window_duration * config.fs)))
    envelope = np.zeros(n_samples)

    if config.signal_window_center is None:
        center_time = config.duration / 2.0
    else:
        center_time = config.signal_window_center

    center_index = int(round(center_time * config.fs))
    start_index = center_index - window_length // 2
    start_index = max(0, min(start_index, n_samples - window_length))
    end_index = start_index + window_length

    # La duree utile du signal est imposee par une fenetre de Hann de taille T.
    envelope[start_index:end_index] = np.hanning(window_length)
    complex_tone *= envelope

    signal_window_start = start_index / config.fs
    signal_window_end = end_index / config.fs

    return Signal(complex_tone, config.fs), fc, target_bin_index, signal_window_start, signal_window_end


def compute_stft(
    signal: Signal,
    stft_parameters: StftParameters,
) -> NSpectrogram:
    return signal.stft(**_get_stft_kwargs(stft_parameters))


def apply_time_shift(signal: Signal, shift_samples: int) -> Signal:
    shifted_data = np.zeros_like(signal.data)
    if shift_samples == 0:
        shifted_data[:] = signal.data
    elif shift_samples > 0:
        shifted_data[shift_samples:] = signal.data[:-shift_samples]
    else:
        advance = -shift_samples
        shifted_data[:-advance] = signal.data[advance:]
    return Signal(shifted_data, signal.freq)

def select_frequency_indices(
    target_bin_index: int,
    n_frequencies: int,
    n_neighbor_bins: int,
) -> np.ndarray:
    start = max(0, target_bin_index - n_neighbor_bins)
    stop = min(n_frequencies, target_bin_index + n_neighbor_bins + 1)
    return np.arange(start, stop, dtype=int)


def select_time_indices(
    spectrogram: NSpectrogram,
    signal_window_start: float,
    signal_window_end: float,
    n_time_slices: int,
) -> np.ndarray:
    stft_times = spectrogram.t
    if stft_times.size == 0:
        raise ValueError("Le spectrogramme ne contient aucune trame temporelle.")

    window_center = 0.5 * (signal_window_start + signal_window_end)
    window_half_duration = 0.5 * max(signal_window_end - signal_window_start, 0.0)

    candidate_times = np.linspace(
        window_center - window_half_duration,
        window_center + window_half_duration,
        max(1, n_time_slices),
    )
    candidate_times = np.clip(candidate_times, stft_times[0], stft_times[-1])
    selected_indices = np.array([np.argmin(np.abs(stft_times - t)) for t in candidate_times], dtype=int)

    return np.unique(selected_indices)


def analyze_complex_tone_stft(
    config: ComplexToneStftStudyConfig | None = None,
) -> ComplexToneStftStudyResult:
    if config is None:
        config = ComplexToneStftStudyConfig()

    if abs(config.time_shift_samples) >= config.stft_parameters.nperseg:
        raise ValueError(
            f"time_shift_samples doit etre strictement inferieur a nperseg={config.stft_parameters.nperseg}."
        )

    signal_s1, fc, target_bin_index, signal_window_start, signal_window_end = generate_complex_tone(config)
    signal_s2 = apply_time_shift(signal_s1, config.time_shift_samples)
    spectrogram_s1 = compute_stft(signal_s1, config.stft_parameters)
    spectrogram_s2 = compute_stft(signal_s2, config.stft_parameters)
    selected_freq_indices = select_frequency_indices(
        target_bin_index=target_bin_index,
        n_frequencies=spectrogram_s1.Sxx.shape[1],
        n_neighbor_bins=config.n_neighbor_bins,
    )
    selected_time_indices_s1 = select_time_indices(
        spectrogram=spectrogram_s1,
        signal_window_start=signal_window_start,
        signal_window_end=signal_window_end,
        n_time_slices=config.n_time_slices,
    )
    selected_time_indices_s2 = select_time_indices(
        spectrogram=spectrogram_s2,
        signal_window_start=signal_window_start + config.time_shift_samples / config.fs,
        signal_window_end=signal_window_end + config.time_shift_samples / config.fs,
        n_time_slices=config.n_time_slices,
    )

    return ComplexToneStftStudyResult(
        config=config,
        signal_s1=signal_s1,
        signal_s2=signal_s2,
        spectrogram_s1=spectrogram_s1,
        spectrogram_s2=spectrogram_s2,
        fc=fc,
        target_bin_index=target_bin_index,
        signal_window_start=signal_window_start,
        signal_window_end=signal_window_end,
        time_shift_seconds=config.time_shift_samples / config.fs,
        selected_freq_indices=selected_freq_indices,
        selected_time_indices_s1=selected_time_indices_s1,
        selected_time_indices_s2=selected_time_indices_s2,
    )


def _get_phase(values: np.ndarray, unwrap_phase: bool) -> np.ndarray:
    phase = np.angle(values)
    if unwrap_phase:
        return np.unwrap(phase)
    return phase


def _get_sorted_frequency_view(spectrogram: NSpectrogram) -> tuple[np.ndarray, np.ndarray]:
    sort_indices = np.argsort(spectrogram.f)
    return spectrogram.f[sort_indices], sort_indices


def _apply_plot_zoom(ax: Axes, zoom: PlotZoom | None) -> None:
    if zoom is None:
        return
    if zoom.xlim is not None:
        ax.set_xlim(*zoom.xlim)
    if zoom.ylim is not None:
        ax.set_ylim(*zoom.ylim)


def _get_signal_and_window(
    result: ComplexToneStftStudyResult,
    signal_name: str,
) -> tuple[Signal, float, float]:
    if signal_name == "s1":
        return result.signal_s1, result.signal_window_start, result.signal_window_end
    if signal_name == "s2":
        return (
            result.signal_s2,
            result.signal_window_start + result.time_shift_seconds,
            result.signal_window_end + result.time_shift_seconds,
        )
    raise ValueError(f"Signal inconnu: {signal_name}")


def _get_spectrogram(
    result: ComplexToneStftStudyResult,
    signal_name: str,
) -> NSpectrogram:
    if signal_name == "s1":
        return result.spectrogram_s1
    if signal_name == "s2":
        return result.spectrogram_s2
    raise ValueError(f"Signal inconnu: {signal_name}")


def _get_selected_time_indices(
    result: ComplexToneStftStudyResult,
    signal_name: str,
) -> np.ndarray:
    if signal_name == "s1":
        return result.selected_time_indices_s1
    if signal_name == "s2":
        return result.selected_time_indices_s2
    raise ValueError(f"Signal inconnu: {signal_name}")


def plot_time_signal(
    result: ComplexToneStftStudyResult,
    signal_name: str,
    ax: Axes | None = None,
    zoom: PlotZoom | None = None,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    signal, window_start, window_end = _get_signal_and_window(result, signal_name)
    time = signal.time
    ax.plot(time, np.real(signal.data), label="Re{x(t)}")
    ax.plot(time, np.abs(signal.data), label="|x(t)|", linestyle="--")
    ax.axvline(window_start, color="k", linestyle=":", linewidth=1.0)
    ax.axvline(window_end, color="k", linestyle=":", linewidth=1.0)
    ax.set_title(f"Signal temporel {signal_name}")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend()
    _apply_plot_zoom(ax, zoom)
    return ax


def plot_phase_vs_time(
    result: ComplexToneStftStudyResult,
    signal_name: str,
    ax: Axes | None = None,
    zoom: PlotZoom | None = None,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    spectrogram = _get_spectrogram(result, signal_name)
    for freq_index in result.selected_freq_indices:
        phase = _get_phase(
            spectrogram.Sxx[0, freq_index, :],
            unwrap_phase=result.config.unwrap_phase,
        )
        ax.plot(spectrogram.t, phase, label=f"f={spectrogram.f[freq_index]:.2f} Hz")

    ax.set_title(f"Argument de la STFT de {signal_name} en fonction du temps")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Phase (rad)")
    ax.grid(True)
    ax.legend()
    _apply_plot_zoom(ax, zoom)
    return ax


def plot_phase_vs_frequency(
    result: ComplexToneStftStudyResult,
    signal_name: str,
    ax: Axes | None = None,
    zoom: PlotZoom | None = None,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    spectrogram = _get_spectrogram(result, signal_name)
    selected_time_indices = _get_selected_time_indices(result, signal_name)
    sorted_frequencies, sort_indices = _get_sorted_frequency_view(spectrogram)
    for time_index in selected_time_indices:
        values = spectrogram.Sxx[0, :, time_index][sort_indices]
        phase = _get_phase(values, unwrap_phase=result.config.unwrap_phase)
        ax.plot(sorted_frequencies, phase, label=f"t={spectrogram.t[time_index]:.4f} s")

    ax.set_title(f"Argument de la STFT de {signal_name} en fonction de la fréquence")
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("Phase (rad)")
    ax.grid(True)
    ax.legend()
    _apply_plot_zoom(ax, zoom)
    return ax


def plot_magnitude_vs_time(
    result: ComplexToneStftStudyResult,
    signal_name: str,
    ax: Axes | None = None,
    zoom: PlotZoom | None = None,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    spectrogram = _get_spectrogram(result, signal_name)
    for freq_index in result.selected_freq_indices:
        magnitude = np.abs(spectrogram.Sxx[0, freq_index, :])
        ax.plot(spectrogram.t, magnitude, label=f"f={spectrogram.f[freq_index]:.2f} Hz")

    ax.set_title(f"Module de la STFT de {signal_name} en fonction du temps")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("|STFT|")
    ax.grid(True)
    ax.legend()
    _apply_plot_zoom(ax, zoom)
    return ax


def plot_magnitude_vs_frequency(
    result: ComplexToneStftStudyResult,
    signal_name: str,
    ax: Axes | None = None,
    zoom: PlotZoom | None = None,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    spectrogram = _get_spectrogram(result, signal_name)
    selected_time_indices = _get_selected_time_indices(result, signal_name)
    sorted_frequencies, sort_indices = _get_sorted_frequency_view(spectrogram)
    for time_index in selected_time_indices:
        magnitude = np.abs(spectrogram.Sxx[0, :, time_index][sort_indices])
        ax.plot(sorted_frequencies, magnitude, label=f"t={spectrogram.t[time_index]:.4f} s")

    ax.set_title(f"Module de la STFT de {signal_name} en fonction de la fréquence")
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("|STFT|")
    ax.grid(True)
    ax.legend()
    _apply_plot_zoom(ax, zoom)
    return ax


def plot_phase_difference_vs_time(
    result: ComplexToneStftStudyResult,
    ax: Axes | None = None,
    zoom: PlotZoom | None = None,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    spectrogram_s1 = result.spectrogram_s1
    spectrogram_s2 = result.spectrogram_s2
    for freq_index in result.selected_freq_indices:
        phase_s1 = _get_phase(spectrogram_s1.Sxx[0, freq_index, :], unwrap_phase=result.config.unwrap_phase)
        phase_s2 = _get_phase(spectrogram_s2.Sxx[0, freq_index, :], unwrap_phase=result.config.unwrap_phase)
        ax.plot(
            spectrogram_s1.t,
            phase_s2 - phase_s1,
            label=f"f={spectrogram_s1.f[freq_index]:.2f} Hz",
        )

    ax.set_title("Difference des arguments arg(S2)-arg(S1) en fonction du temps")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Phase (rad)")
    ax.grid(True)
    ax.legend()
    _apply_plot_zoom(ax, zoom)
    return ax


def plot_magnitude_ratio_vs_time(
    result: ComplexToneStftStudyResult,
    ax: Axes | None = None,
    zoom: PlotZoom | None = None,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    eps = 1e-12
    relative_threshold = 1e-3
    line_styles = ["-", "--", "-.", ":"]
    spectrogram_s1 = result.spectrogram_s1
    spectrogram_s2 = result.spectrogram_s2
    for curve_idx, freq_index in enumerate(result.selected_freq_indices):
        magnitude_s1 = np.abs(spectrogram_s1.Sxx[0, freq_index, :])
        magnitude_s2 = np.abs(spectrogram_s2.Sxx[0, freq_index, :])
        valid = magnitude_s1 > relative_threshold * np.max(magnitude_s1)
        ratio = np.full_like(magnitude_s1, np.nan, dtype=float)
        ratio[valid] = magnitude_s2[valid] / np.maximum(magnitude_s1[valid], eps)
        ax.plot(
            spectrogram_s1.t,
            ratio,
            label=f"f={spectrogram_s1.f[freq_index]:.2f} Hz",
            linestyle=line_styles[curve_idx % len(line_styles)],
        )

    ax.set_title("Rapport des modules |S2| / |S1| en fonction du temps")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("|S2| / |S1|")
    ax.grid(True)
    ax.legend()
    _apply_plot_zoom(ax, zoom)
    return ax


def plot_all_study_figures(
    result: ComplexToneStftStudyResult,
    zoom: StudyPlotZoom | None = None,
) -> tuple[Figure, np.ndarray]:
    fig, axes = plt.subplots(6, 2, figsize=(14, 24))
    if zoom is None:
        zoom = StudyPlotZoom()

    plot_time_signal(result, "s1", ax=axes[0, 0], zoom=zoom.time_signal_s1)
    plot_time_signal(result, "s2", ax=axes[0, 1], zoom=zoom.time_signal_s2)
    plot_phase_vs_time(result, "s1", ax=axes[1, 0], zoom=zoom.phase_vs_time_s1)
    plot_phase_vs_frequency(result, "s1", ax=axes[1, 1], zoom=zoom.phase_vs_frequency_s1)
    plot_magnitude_vs_time(result, "s1", ax=axes[2, 0], zoom=zoom.magnitude_vs_time_s1)
    plot_magnitude_vs_frequency(result, "s1", ax=axes[2, 1], zoom=zoom.magnitude_vs_frequency_s1)
    plot_phase_vs_time(result, "s2", ax=axes[3, 0], zoom=zoom.phase_vs_time_s2)
    plot_phase_vs_frequency(result, "s2", ax=axes[3, 1], zoom=zoom.phase_vs_frequency_s2)
    plot_magnitude_vs_time(result, "s2", ax=axes[4, 0], zoom=zoom.magnitude_vs_time_s2)
    plot_magnitude_vs_frequency(result, "s2", ax=axes[4, 1], zoom=zoom.magnitude_vs_frequency_s2)
    plot_phase_difference_vs_time(result, ax=axes[5, 0], zoom=zoom.phase_difference_vs_time)
    plot_magnitude_ratio_vs_time(result, ax=axes[5, 1], zoom=zoom.magnitude_ratio_vs_time)

    active_duration = result.signal_window_end - result.signal_window_start
    fig.suptitle(
        "Etude STFT d'une exponentielle complexe "
        f"(f_c={result.fc:.2f} Hz, bin={result.target_bin_index}, "
        f"T={active_duration:.3f} s, retard={result.time_shift_seconds:.6f} s)"
    )
    fig.tight_layout()
    return fig, axes


def run_demo(
    config: ComplexToneStftStudyConfig | None = None,
    zoom: StudyPlotZoom | None = None,
) -> ComplexToneStftStudyResult:
    result = analyze_complex_tone_stft(config)
    plot_all_study_figures(result, zoom=zoom)
    plt.show()
    return result


if __name__ == "__main__":
    run_demo()
