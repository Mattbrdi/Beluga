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
from ..utils.sub_classes import AudioArray, Tetrahedra, Environment
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
