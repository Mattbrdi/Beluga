import numpy as np
from scipy.signal import stft
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray


def compute_waveform_derivative(
    waveform: NDArray[np.float64],
    frame_rate: float | None = None,
) -> NDArray[np.float64]:
    waveform = np.asarray(waveform, dtype=np.float64)
    if waveform.shape[-1] == 0:
        return waveform.copy()
    if waveform.shape[-1] == 1:
        return np.zeros_like(waveform)
    if frame_rate is not None and frame_rate <= 0:
        raise ValueError(f"frame_rate must be positive, got {frame_rate}")

    spacing = 1.0 / frame_rate if frame_rate else 1.0
    return np.gradient(waveform, spacing, axis=-1)


def _mask_intervals(mask_1d: NDArray[np.bool_]) -> list[tuple[int, int]]:
    mask_1d = np.asarray(mask_1d, dtype=bool)
    padded = np.r_[False, mask_1d, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [tuple(interval) for interval in changes.reshape(-1, 2)]


def _weighted_frequency_stats(
    signal_1d: NDArray[np.float64],
    frame_rate: float,
    start_index: int,
    end_index: int,
    fmin: float,
    fmax: float,
    n_fft: int,
    hop_length: int,
    energy_floor: float,
) -> dict[str, float]:
    nperseg = min(n_fft, signal_1d.size)
    if nperseg < 2:
        return {
            "mean_freq": np.nan,
            "freq_variance": np.nan,
            "freq_std": np.nan,
            "energy": 0.0,
        }

    hop_length = min(hop_length, nperseg)
    noverlap = nperseg - hop_length
    freqs, times, Zxx = stft(
        signal_1d,
        fs=frame_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=n_fft,
        detrend=False,
        return_onesided=True,
        boundary=None,
        padded=False,
    )

    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(freq_mask) or times.size == 0:
        return {
            "mean_freq": np.nan,
            "freq_variance": np.nan,
            "freq_std": np.nan,
            "energy": 0.0,
        }

    start_time = start_index / frame_rate
    end_time = end_index / frame_rate
    half_window = nperseg / (2 * frame_rate)
    time_mask = (times + half_window > start_time) & (times - half_window < end_time)
    if not np.any(time_mask):
        closest_frame = np.argmin(np.abs(times - ((start_time + end_time) / 2)))
        time_mask[closest_frame] = True

    power = np.abs(Zxx[np.ix_(freq_mask, time_mask)]) ** 2
    energy = float(np.sum(power))
    if energy <= energy_floor:
        return {
            "mean_freq": np.nan,
            "freq_variance": np.nan,
            "freq_std": np.nan,
            "energy": energy,
        }

    freqs_band = freqs[freq_mask, np.newaxis]
    mean_freq = float(np.sum(freqs_band * power) / energy)
    freq_variance = float(np.sum(((freqs_band - mean_freq) ** 2) * power) / energy)

    return {
        "mean_freq": mean_freq,
        "freq_variance": freq_variance,
        "freq_std": float(np.sqrt(freq_variance)),
        "energy": energy,
    }


def filter_whistle_false_positives(
    input_signal: NDArray[np.float64],
    frame_rate: float,
    impulsive_mask: NDArray[np.uint8],
    fmin: float = 500.0,
    fmax: float = 20_000.0,
    mean_freq_threshold: float = 2_500.0,
    freq_std_threshold: float  = 300.0,
    min_blob_duration: float = 0.0,
    n_fft: int = 4096,
    hop_length: int = 1024,
    energy_floor: float = 1e-20,
    return_blob_metrics: bool = False,
):
    """Remove narrowband, high-frequency blobs from a sample-domain mask.

    Each contiguous 1-region in ``impulsive_mask`` is treated as one candidate
    blob. The blob is rejected when its STFT energy inside ``[fmin, fmax]`` is
    centered above ``mean_freq_threshold`` and its weighted frequency spread is
    below the configured variance or standard-deviation threshold.
    """

    input_signal = np.asarray(input_signal, dtype=np.float64)
    impulsive_mask = np.asarray(impulsive_mask)
    if input_signal.shape != impulsive_mask.shape:
        raise ValueError(
            "input_signal and impulsive_mask must have the same shape. "
            f"Got {input_signal.shape} and {impulsive_mask.shape}."
        )

    if input_signal.shape[-1] == 0:
        return (impulsive_mask.copy(), []) if return_blob_metrics else impulsive_mask.copy()

    filtered_mask = impulsive_mask.copy()
    leading_shape = input_signal.shape[:-1]
    flat_signal = input_signal.reshape(-1, input_signal.shape[-1])
    flat_mask = filtered_mask.reshape(-1, filtered_mask.shape[-1])
    metrics = []

    for row_index, (signal_1d, mask_1d) in enumerate(zip(flat_signal, flat_mask)):
        leading_index = tuple(int(index) for index in np.unravel_index(row_index, leading_shape)) if leading_shape else ()
        for start_index, end_index in _mask_intervals(mask_1d):
            duration = (end_index - start_index) / frame_rate
            blob_stats = _weighted_frequency_stats(
                signal_1d,
                frame_rate,
                start_index,
                end_index,
                fmin,
                fmax,
                n_fft,
                hop_length,
                energy_floor,
            )

            spread = blob_stats["freq_std"]
            spread_threshold = freq_std_threshold

            rejected = (
                duration >= min_blob_duration
                and blob_stats["mean_freq"] >= mean_freq_threshold
                and spread <= spread_threshold
            )
            if rejected:
                mask_1d[start_index:end_index] = 0

            metrics.append(
                {
                    "leading_index": leading_index,
                    "start_index": int(start_index),
                    "end_index": int(end_index),
                    "duration": float(duration),
                    "mean_freq": blob_stats["mean_freq"],
                    "freq_variance": blob_stats["freq_variance"],
                    "freq_std": blob_stats["freq_std"],
                    "energy": blob_stats["energy"],
                    "rejected": bool(rejected),
                }
            )

    if return_blob_metrics:
        return filtered_mask, metrics
    return filtered_mask


def threshold_model(input_signal : NDArray[np.float64],
                    frame_rate,
                    pulse_duration = 0.001,
                    pulse_overlap = 0.000675,
                    z_threshold=3,
                    local_pulse_radius=50,
                    ):
    if not 0 <= pulse_duration <= 1:
        raise ValueError(f"Wrong pulse_duration value. Got {pulse_duration}, expecting value between 0 and 1s")

    if not 0 <= pulse_overlap <= pulse_duration:
        raise ValueError(f"Wrong pulse_overlap value. Got {pulse_overlap}, expecting value between 0 and pulse_duration")

    input_signal = np.asarray(input_signal, dtype=np.float64)
    N = np.shape(input_signal)[-1]
    num_index_pulse_duration = int(round(pulse_duration * frame_rate))
    num_index_pulse_overlap = int(round(pulse_overlap * frame_rate))

    if num_index_pulse_duration <= 0:
        raise ValueError(
            f"pulse_duration={pulse_duration} is too short for frame_rate={frame_rate}. "
            "It must contain at least one sample."
        )

    if N < num_index_pulse_duration:
        raise ValueError(
            f"Signal has {N} samples, but pulse_duration needs "
            f"{num_index_pulse_duration} samples."
        )

    num_step = num_index_pulse_duration - num_index_pulse_overlap
    if num_step <= 0:
        raise ValueError("pulse_overlap must be smaller than pulse_duration")

    if local_pulse_radius < 0:
        raise ValueError(f"local_pulse_radius must be >= 0. Got {local_pulse_radius}")

    starts = np.arange(0, N - num_index_pulse_duration + 1, num_step, dtype=int)
    if starts[-1] != N - num_index_pulse_duration:
        starts = np.append(starts, N - num_index_pulse_duration)

    pulses = sliding_window_view(
        input_signal,
        window_shape=num_index_pulse_duration,
        axis=-1,
    )[..., starts, :]

    pulse_energy = np.mean(pulses * pulses, axis=-1)
    pulse_outlier_mask = np.zeros_like(pulse_energy, dtype=bool)

    for pulse_index in range(starts.size):
        local_min = max(0, pulse_index - local_pulse_radius)
        local_max = min(starts.size, pulse_index + local_pulse_radius + 1)
        local_energy = pulse_energy[..., local_min:local_max]

        median = np.median(local_energy, axis=-1)
        mad = np.median(np.abs(local_energy - median[..., np.newaxis]), axis=-1)
        std = np.maximum(1.4826 * mad, 1e-12)
        z_score = (pulse_energy[..., pulse_index] - median) / std

        pulse_outlier_mask[..., pulse_index] = z_score > z_threshold

    impulsive_mask = np.zeros_like(input_signal, dtype=np.uint8)
    for pulse_index, start in enumerate(starts):
        pulse_is_impulsive = pulse_outlier_mask[..., pulse_index]
        impulsive_mask[..., start:start + num_index_pulse_duration] = np.maximum(
            impulsive_mask[..., start:start + num_index_pulse_duration],
            pulse_is_impulsive[..., np.newaxis].astype(np.uint8),
        )

    return impulsive_mask
