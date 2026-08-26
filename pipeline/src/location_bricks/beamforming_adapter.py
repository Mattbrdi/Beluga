import sys
from pathlib import Path


import numpy as np
from numpy.typing import NDArray
from src.utils.sub_classes import (
    Tetrahedra,
    AudioArray,
    BeamformingParameters,
    TimeFrequencyMaskParameters,
)
from beamforming.geometry import TetrahedralArray, Direction
from beamforming.beamformers.music import MUSIC
from beamforming.beamformers.mvdr import MVDR
from beamforming.workflows.narrowband.basenarrowband import NarrowbandBeamformer
from beamforming.workflows.wideband.cssm import CSSM
from beamforming.workflows.wideband.issm import ISSM
from beamforming.workflows.wideband.srp_phat import SRPPHAT
from beamforming.workflows.wideband.tops import TOPS
from beamforming.workflows.wideband.masked import MaskedBeamformer
from beamforming.grid import SearchGrid
from beamforming.signal.stft import compute_band_stft, STFTConfig
from time_frequency_mask.masknet.run_inference import (
    get_mask_from_array_arbitrary_size,
)
from time_frequency_mask.tdoa_estimation.blob import (
    output_blobs_from_mask,
    output_mask_from_blobs,
    blob_filtering_heuristic,
)

# TODO: Fix this by removing the need for sys and pathlib
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# turn tetrahedra of pipeline to tetrahedra of beamforming
def get_tetrahedra(pipeline_tetrahedra: Tetrahedra) -> TetrahedralArray:
    """Convert a pipeline Tetrahedra into a TetrahedralArray for beamforming operations

    Parameters
    ----------
    pipeline_tetrahedra : Tetrahedra
        A tetrahedra instance of the Tetrahedra class in the pipeline

    Returns
    -------
    TetrahedralArray
        Array geometry in meters and in the enu referential
    """
    pos = pipeline_tetrahedra.rotated_hydro_pos_enu
    return TetrahedralArray(pos)


def perform_doa(
    tetrahedra,
    audio_array,
    grid,
    workflow,
    use_tf_mask,
    *,
    central_frequency=None,
    beamformer=None,
    Zxx=None,
    freqs=None,
    mask=None,
    blobs=None,
    initial_doa=None,
) -> Direction:
    if workflow == "narrowband":
        if use_tf_mask:
            doa_estimator = MaskedBeamformer(tetrahedra, beamformer).compute(
                audio_array, grid, blobs
            )
        else:
            doa_estimator = NarrowbandBeamformer(tetrahedra, beamformer).compute(
                audio_array, grid, central_frequency
            )
    elif workflow == "ISSM":
        doa_estimator = ISSM(tetrahedra, beamformer, None).compute_from_stft(
            Zxx, freqs, grid
        )
    elif workflow == "CSSM":
        doa_estimator = CSSM(
            tetrahedra, beamformer, None, central_frequency, initial_doa
        ).compute_from_stft(Zxx, freqs, grid)
    elif workflow == "SRPPHAT":
        if use_tf_mask:
            weights = mask != 0
            doa_estimator = SRPPHAT(tetrahedra, None).compute_weighted_from_stft(
                Zxx, freqs, grid, weights
            )
        else:
            doa_estimator = SRPPHAT(tetrahedra, None).compute_from_stft(
                Zxx, freqs, grid
            )
    elif workflow == "TOPS":
        ref_freq_idx = np.argmin(np.abs(freqs - central_frequency))
        doa_estimator = TOPS(tetrahedra, None, 1, ref_freq_idx).compute_from_stft(
            Zxx, freqs, grid
        )
    else:
        raise ValueError(
            f"Provided beamforming method doesn't exist or isn't implemented yet, got workflow: {workflow} instead of either narrowband, ISSM, CSSM, SRPPHAT, TOPS"
        )

    return doa_estimator.doa


def beamforming_doa(
    beamforming_parameters: BeamformingParameters,
    central_frequency: float,
    tetrahedra: TetrahedralArray,
    audio_array: AudioArray,
    sound_speed: float,
    stft_config: STFTConfig = STFTConfig(),
    tf_mask_parameters: TimeFrequencyMaskParameters | None = None,
    tf_mask_model=None,
):
    if tf_mask_parameters is None:
        tf_mask_parameters = TimeFrequencyMaskParameters()
    audio_array_data = audio_array.data_array
    sampling_rate = audio_array.metadata.sample_rate

    beamformer_method = beamforming_parameters.beamformer
    workflow = beamforming_parameters.workflow
    use_tf_mask = tf_mask_parameters.use_tf_mask
    mesh_size = beamforming_parameters.mesh_size
    use_coarse_and_fine_search = beamforming_parameters.use_coarse_and_fine_search

    beamformer, freqs, Zxx = None, None, None
    initial_doa, coarse_grid, fine_grid = None, None, None
    blobs, filtered_blobs, filtered_mask = None, None, None

    if beamformer_method == "mvdr":
        beamformer = MVDR()
    elif beamformer_method == "music":
        beamformer = MUSIC(1)
    else:
        raise ValueError(
            f"Provided beamformer is not implemented yet or does not exist. Got {beamformer_method} instead of mvdr or music"
        )        

    tf_params = tf_mask_parameters.tf_parameters

    if use_tf_mask:
        audio, stft = tf_params.audio, tf_params.stft
        stft_config = STFTConfig(
            n_fft=stft.n_fft,
            hop_length=stft.hop_length,
            window=stft.window,
            boundary=stft.boundary,
            padded=stft.padded,
            min_freq=audio.min_freq,
            max_freq=audio.max_freq
        )

    if use_tf_mask:
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
        filtered_mask = output_mask_from_blobs(filtered_blobs, *original_mask.shape)

    if workflow != "narrowband":
        freqs, _, Zxx = compute_band_stft(audio_array_data, stft_config, sampling_rate)

        if use_tf_mask:
            Zxx = (filtered_mask != 0)[:, None, :] * Zxx

    delta_beam = sound_speed / (central_frequency * tetrahedra.D)  # Lobe ambiguity size
    if use_coarse_and_fine_search:
        size = int(max(np.ceil(8 * np.pi / delta_beam), mesh_size / 10))
        coarse_grid = SearchGrid.full_sphere(
            n_theta=size,
            n_phi=size,
        )
    else:
        coarse_grid = SearchGrid.full_sphere(
            n_theta=mesh_size,
            n_phi=mesh_size,
        )

    if workflow == "CSSM":
        # initial_doa = ISSM(tetrahedra, beamformer, None).compute_from_stft(Zxx, freqs, coarse_grid).doa
        initial_doa = (
            SRPPHAT(tetrahedra, None).compute_from_stft(Zxx, freqs, coarse_grid).doa
        )

    if workflow in ["TOPS", "ISSM"] and use_tf_mask:
        valid_freqs = np.any(filtered_mask != 0, axis=1)
        Zxx = Zxx[valid_freqs]
        freqs = freqs[valid_freqs]

    doa = perform_doa(
        tetrahedra,
        audio_array_data,
        coarse_grid,
        workflow,
        use_tf_mask,
        central_frequency=central_frequency,
        beamformer=beamformer,
        Zxx=Zxx,
        freqs=freqs,
        mask=filtered_mask,
        blobs=filtered_blobs,
        initial_doa=initial_doa,
    )

    if use_coarse_and_fine_search:
        fine_grid = SearchGrid.around(
            doa,
            2 * delta_beam,
            n_theta=mesh_size,
            n_phi=mesh_size,
        )
        doa = perform_doa(
            tetrahedra,
            audio_array_data,
            fine_grid,
            workflow,
            use_tf_mask,
            central_frequency=central_frequency,
            beamformer=beamformer,
            Zxx=Zxx,
            freqs=freqs,
            mask=filtered_mask,
            blobs=filtered_blobs,
            initial_doa=initial_doa,
        )

    return doa.vector
