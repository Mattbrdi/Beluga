import numpy as np
from numpy.typing import NDArray

from pathlib import Path
import sys

# TODO: Fix this by removing the need for sys and pathlib
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from time_frequency_mask.config import Parameters as TFParameters
from time_frequency_mask.config import STFTParamters
from time_frequency_mask.masknet.run_inference import (
    load_model,
    get_mask_from_array_arbitrary_size,
)
from time_frequency_mask.masknet.models.spectro_mask_net import SpectroMaskNet
from time_frequency_mask.tdoa_estimation.blob import (
    output_blobs_from_mask,
    blob_filtering_heuristic,
    output_mask_from_blobs,
)
from ..utils.sub_classes import Parameters as PL_Parameters
from ..utils.sub_classes import (
    AudioArray,
    Environment,
    Tetrahedra,
    TimeFrequencyMaskParameters,
)
from time_frequency_mask.stft import scipy_stft_complex_psd, frequency_band
from time_frequency_mask.data_generation.core.power_computation import compute_P_moy

def tf_mask_compatibility(
    audio_array: AudioArray,
    environment: Environment,
    tf_parameters: TFParameters,
):
    if tf_parameters is None:
        raise ValueError("TF-mask parameters are missing")

    num_mics = audio_array.data_array.shape[0]
    if num_mics != tf_parameters.array.num_mics:
        raise ValueError(
            f"TF-mask expects {tf_parameters.array.num_mics} microphones, "
            f"pipeline audio has {num_mics}"
        )

    if audio_array.metadata.sample_rate != tf_parameters.audio.sampling_rate:
        raise ValueError(
            f"TF-mask expects {tf_parameters.audio.sampling_rate} Hz, "
            f"pipeline audio uses {audio_array.metadata.sample_rate} Hz"
        )

    tetrahedra = environment.tetrahedras[audio_array.metadata.tetra_id]
    if not np.isclose(
        tetrahedra.max_delay_seconds,
        tf_parameters.array.max_tdoa,
        rtol=0.1,
    ):
        raise ValueError(
            f"TF-mask max TDOA is {tf_parameters.array.max_tdoa}, "
            f"pipeline max delay is {tetrahedra.max_delay_seconds}"
        )

    if not np.isclose(
        environment.sound_speed,
        tf_parameters.array.sound_speed,
    ):
        raise ValueError(
            f"TF-mask sound speed is {tf_parameters.array.sound_speed}, "
            f"pipeline sound speed is {environment.sound_speed}"
        )

def load_tf_mask_model(
    tf_parameters: TFParameters, use_phase_aware_network: bool
) -> SpectroMaskNet:
    M = tf_parameters.array.num_mics

    if use_phase_aware_network:
        model = SpectroMaskNet(n_channels=M * M)
    else:
        model = SpectroMaskNet(n_channels=M)

    return load_model(model, tf_parameters.network.checkpoint_path)


def tf_frequency_band_mask(tf_parameters: TFParameters) -> NDArray[np.bool_]:
    stft = tf_parameters.stft
    audio = tf_parameters.audio
    freqs = np.fft.rfftfreq(stft.n_fft, d=1.0 / audio.sampling_rate)
    return (freqs >= audio.min_freq) & (freqs <= audio.max_freq)


def compute_filtered_tf_mask(
    audio_array: AudioArray,
    tf_mask_parameters: TimeFrequencyMaskParameters,
    tf_mask_model,
) -> NDArray[np.bool_]:
    if not tf_mask_parameters.use_tf_mask:
        raise ValueError("TF-mask is disabled")
    if tf_mask_parameters.tf_parameters is None:
        raise ValueError("TF-mask parameters are missing")

    tf_params = tf_mask_parameters.tf_parameters
    image_size = tf_params.network.image_size
    original_mask = get_mask_from_array_arbitrary_size(
        audio_array.data_array,
        tf_mask_model,
        tf_params,
        image_size,
        image_size // 2,
    )
    blobs = output_blobs_from_mask(original_mask, tf_params)
    filtered_blobs = blob_filtering_heuristic(blobs, tf_params.audio.min_freq)
    return output_mask_from_blobs(filtered_blobs, *original_mask.shape)


def expand_band_mask_to_full_stft(
    mask: NDArray[np.bool_],
    tf_parameters: TFParameters,
) -> NDArray[np.bool_]:
    mask = np.asarray(mask, dtype=bool)
    frequency_band = tf_frequency_band_mask(tf_parameters)
    expected_n_freqs = int(np.sum(frequency_band))
    if mask.ndim != 2 or mask.shape[0] != expected_n_freqs:
        raise ValueError(
            f"Expected a band-limited mask with {expected_n_freqs} frequencies, got {mask.shape}"
        )

    full_mask = np.zeros((frequency_band.size, mask.shape[1]), dtype=bool)
    full_mask[frequency_band, :] = mask
    return full_mask


def extract_band_mask_from_full_stft(
    mask: NDArray[np.bool_],
    tf_parameters: TFParameters,
) -> NDArray[np.bool_]:
    mask = np.asarray(mask, dtype=bool)
    frequency_band = tf_frequency_band_mask(tf_parameters)
    if mask.ndim != 2 or mask.shape[0] != frequency_band.size:
        raise ValueError(
            f"Expected a full STFT mask with {frequency_band.size} frequencies, got {mask.shape}"
        )
    return mask[frequency_band, :]


def get_filtered_snrs(
    audio_one: NDArray[np.float64],
    audio_two: NDArray[np.float64],
    mask : NDArray[np.bool_],
    tf_parameters: TFParameters,
) -> tuple[float, float]:
    audio, stft = tf_parameters.audio, tf_parameters.stft
    freqs, _, Zxx_1 = scipy_stft_complex_psd(audio_one, audio.sampling_rate, stft)
    freqs, Zxx_1 = frequency_band(freqs, Zxx_1, audio.min_freq, audio.max_freq)

    df = audio.sampling_rate / stft.n_fft


    noise_mask = mask == 0
    signal_mask = mask == 1

    Pxx_1 = np.abs(Zxx_1) ** 2
    global_noise_psd_1 = np.sum(2 * df * Pxx_1[noise_mask]) / np.sum(
        2 * df * (noise_mask)
    )

    freqs, _, Zxx_2 = scipy_stft_complex_psd(audio_two, audio.sampling_rate, stft)
    freqs, Zxx_2 = frequency_band(freqs, Zxx_2, audio.min_freq, audio.max_freq)

    Pxx_2 = np.abs(Zxx_2) ** 2
    global_noise_psd_2 = np.sum(2 * df * Pxx_2[noise_mask]) / np.sum(
        2 * df * (noise_mask)
    )

    signal_1_mean_power, noise_1_mean_power = compute_P_moy(
        Pxx_1, signal_mask, global_noise_psd_1, df=df
    )
    signal_2_mean_power, noise_2_mean_power = compute_P_moy(
        Pxx_2, signal_mask, global_noise_psd_2, df=df
    )

    snr_ref, snr_delta = None, None
    if noise_1_mean_power > 1e-12:
        snr_ref = signal_1_mean_power / noise_1_mean_power
    
    if noise_2_mean_power > 1e-12:
        snr_delta = signal_2_mean_power / noise_2_mean_power

    return snr_ref, snr_delta
