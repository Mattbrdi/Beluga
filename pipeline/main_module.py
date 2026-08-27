"""Main function modules to go from wav to positions."""

import os
from datetime import timedelta
from time import time
import numpy as np
import soundfile as sf
import pandas as pd
from src.detection_bricks.mono_audio_detection import SpectrogramGenerator, MobileNetMultilabel, get_audio_start_time, run_pipeline_overlaps_long_spects
from src.detection_bricks.canals_matching import spotting_to_location_preparation
from src.detection_bricks.tf_mask_adapter import (
    compute_filtered_tf_mask,
    expand_band_mask_to_full_stft,
    extract_band_mask_from_full_stft,
    load_tf_mask_model,
    tf_mask_compatibility,
)
from src.utils.sub_classes import Environment, Parameters, AudioMetadata, AudioArray
from src.location_bricks.frequencies_filtering import filter_audio_array, filter_audio_array_from_calltype
from src.denoising_bricks.vmd_denoising import vmd_denoise
from src.location_bricks.tdoa_brick import tdoas
from src.location_bricks.low_level_fusion import low_fusion
from src.location_bricks.high_level_fusion import high_fusion, fusion_bf
from src.location_bricks.beamforming_adapter import get_tetrahedra, beamforming_doa
from src.utils.plots import plot_spectro
from src.denoising_bricks.wt_denoising import wt_denoise


from src.tests.debug_functions import plot_analysis
from scipy.signal import find_peaks, windows

#############################################
########## Detection to AudioArray ##########
#############################################

    
def signal_characteristics(audio_arrays : list[AudioArray], band_threshold_db = -15, frequency_pad = 100, peaks_threshold = 0.5):

    ##### Band selection #####
    if audio_arrays[0].metadata.beluga_call_type == 'Whistle':
        call_band = (500, 25000)
    else :
        call_band = (25000, 190000)
    
    powers = []
    all_freqs = []
    
    for audio_array in audio_arrays:
            
        ##### Compute FFT #####
        n_canaux, n_points = np.shape(audio_array.data_array)
        window = windows.hann(n_points)[None, :]
        signals_win = audio_array.data_array * window
        spectrum = np.fft.rfft(signals_win, axis=1)
        power = np.abs(spectrum)**2
    
        freqs = np.fft.rfftfreq(n_points, 1.0/audio_arrays[0].metadata.sample_rate)
        start_idx = np.searchsorted(freqs, call_band[0])
        end_idx = np.searchsorted(freqs, call_band[1])
        
        power_band = power[:, start_idx:end_idx]
        freqs_band = freqs[start_idx:end_idx]
        
        
        all_freqs.append(freqs_band)
        powers.append(power_band)
    
    central_frequencies = []
    lower_bound_frequencies = []
    upper_bound_frequencies = []
    
    for i in range(len(audio_arrays)):
        for ch in range(np.shape(audio_arrays[i].data_array)[0]):
            p = powers[i][ch]
            f = all_freqs[i]

            """##### Peak detection #####
            peaks, _ = find_peaks(p, height=np.max(p) * peaks_threshold)
            peaks_frequencies = f[peaks]
            
            nb_harmoniques = len(peaks)
            bande_utile_peaks = (peaks_frequencies[0], peaks_frequencies[-1])"""
            
            ##### Useful band #####
            threshold = np.max(p) * (10**(band_threshold_db/10))
            indices = np.where(p >= threshold)[0]
            lower_bound_frequencies.append(f[indices[0]])
            upper_bound_frequencies.append(f[indices[-1]])

            ##### Central frequency #####
            central_frequencies.append(np.sum(f * p) / np.sum(p))

    if audio_arrays[0].metadata.beluga_call_type == 'Whistle':
        limit_band = (500, 25000)
    elif audio_arrays[0].metadata.beluga_call_type == 'HFPC':
        limit_band = (40000, 150000)
    elif audio_arrays[0].metadata.beluga_call_type == 'EC':
        limit_band = (30000, 140000)
    else:
        limit_band = (25000, 170000)

    median_band = (max(limit_band[0], np.median(lower_bound_frequencies)), min(limit_band[1],np.median(upper_bound_frequencies)))
    #TODO
    #band_mask = (f >= bande_utile_peaks[0]) & (f <= bande_utile_peaks[1]) ??
    
    central_frequency = np.median(central_frequencies)
    
    snrs_list = []
    
    for i in range(len(audio_arrays)):
        snrs = []
        for ch in range(np.shape(audio_arrays[i].data_array)[0]):
            
            freqs_band = all_freqs[i]
    
            band_mask = (freqs_band >= median_band[0]) & (freqs_band <= median_band[1])
            p = powers[i][ch]
            
            signal_power = np.sum(p[band_mask]) / np.sum(band_mask)
            noise_power = np.sum(p[~band_mask]) / np.sum(~band_mask)
            snr_power = signal_power / noise_power
            ############################################################################################
            #snr_power /= 10**(5./10) #Removing 14 dB #TODO, aka 3 harmonics
            ############################################################################################
            snrs.append(snr_power)
        snrs_list.append(snrs)
        
    frequency_range = (median_band[0] - frequency_pad , median_band[1] + frequency_pad)

    return central_frequency, frequency_range, snrs_list


def setup_to_detection(
        parameters : Parameters,
        audio_files : list[str],
        beluga_sounds :list[pd.Series],
        call_type : str,
        offset : timedelta,
        environment : Environment,
        sound_mask : np.ndarray
        ):
    
    ##### Process outputs of detection #####
    offset = timedelta(0)
    xdurations, durations, ref_start_times, ref_seconds_since_file_starts = [], [], [], []

    #csv_path = r".\test_data\results\beluga_sounds_debug_qat.csv"
    #df_debug = pd.DataFrame(beluga_sounds)
    #df_debug["call_type"] = call_type
    #df_debug["offset_s"] = offset.total_seconds()
    #write_header = not os.path.exists(csv_path)
    #df_debug.to_csv(csv_path, mode="a", header=write_header, index=False)
    #if parameters.print_level > 0:
    #    print(f"[DEBUG] wrote {len(df_debug)} rows to {csv_path}")
        
    for sound in beluga_sounds:
        xduration = timedelta(seconds=sound['snr_w_range_whistle_duration'])
        xdurations.append(xduration)
        duration = xduration.total_seconds()
        durations.append(duration)
        ref_start_time = sound['Timestamp'] + timedelta(seconds=sound['snr_w_range_whistle_start'])
        ref_start_times.append(ref_start_time)
        ref_seconds_since_file_start = timedelta(
            seconds=float(sound["seconds_since_file_start"]) + float(sound["snr_w_range_whistle_start"])
        )
        ref_seconds_since_file_starts.append(ref_seconds_since_file_start)
        event_start_dt = max(ref_start_times)
        event_end_dt   = min([st + timedelta(seconds=d) for st, d in zip(ref_start_times, durations)])
        event_duration = (event_end_dt - event_start_dt).total_seconds()
        if event_duration <= 0:
            return None, None, None
    
    ##### Debug the outputs #####
    if parameters.print_level > 1:
        print(f"Durée du sifflement: {durations}")
        print(f"Début de T1: {ref_start_times}")
        print(f"seconds_since_file_start: {ref_seconds_since_file_starts}")
   
    ##### Match the audios #####
    pad = np.max(durations)
    sliced_audios, sample_rates = spotting_to_location_preparation(audio_files, ref_seconds_since_file_starts, pad, parameters.print_level)
    
    if parameters.print_level > 1 :
        print(f"seconds_since_file_start_T1: {ref_seconds_since_file_starts[0]}")
        print(f"Shape des sliced_audios : {sliced_audios[0].shape}, {sliced_audios[1].shape}")

    if sliced_audios[0].shape[0] == 0 or sliced_audios[1].shape == 0:
        return None, None, None
    
    ##### Cut the overduration of sliced_audios #####
    #shortened_audios = [select_plage_audio(audio, sample_rate, delay_from_start, duration) for audio, sample_rate, delay_from_start in zip(sliced_audios, sample_rates, ref_start_times)]
    
    if parameters.print_level > 1:
        print(f"Nouveaux débuts: {ref_start_times}") # Size M-1 where M = len(Tetrahedras)
    
    ##### Create the AudioArrays #####
    audio_arrays = []
    for tetrahedra, start_time, audio_data, sample_rate, sound_bool, duration in zip(environment.tetrahedras.values(), ref_start_times, sliced_audios, sample_rates, sound_mask, durations):
        if sound_bool:
            metadata = AudioMetadata(tetrahedra.id, call_type, duration, start_time,  None, sample_rate, None, central_frequency = None)
            # Central frequency and snr require filtration
            audio_array = AudioArray(metadata, tetrahedra, parameters.location_parameters.use_h4, data_array=audio_data.T)
            audio_arrays.append(audio_array)

    if parameters.tf_mask_parameters.use_tf_mask:
        if not audio_arrays:
            raise ValueError("No active audio arrays to validate")
        
        for audio_array in audio_arrays:
            tf_mask_compatibility(
                audio_array,
                environment,
                parameters.tf_mask_parameters.tf_parameters,
            )

    return audio_arrays, event_duration, event_start_dt

def tdoas_mask_check(tdoas_mask : list[np.ndarray]):
    # Expects a list of tdoas ordered as 
    # H1H2 H1H3 H1H4 H2H3 H2H4 H3H4
    sum_bools = [np.sum(tdoa) for tdoa in tdoas_mask]
    usable_tetras = np.sum([sum_bool > 2 for sum_bool in sum_bools])
    if usable_tetras < 2:
        return False
    number_of_h4 = [int(tdoa[2]) + int(tdoa[4]) + int(tdoa[5]) for tdoa in tdoas_mask] # int(True) = 1
    if np.sum(number_of_h4) < 2:
        return False
    h1_used = np.sum([tdoa[0] or tdoa[1] or tdoa[2] for tdoa in tdoas_mask])
    h2_used = np.sum([tdoa[0] or tdoa[3] or tdoa[4] for tdoa in tdoas_mask])
    h3_used = np.sum([tdoa[1] or tdoa[3] or tdoa[5] for tdoa in tdoas_mask])
    h4_used = np.sum([tdoa[2] or tdoa[4] or tdoa[5] for tdoa in tdoas_mask])
    if h1_used + h2_used + h3_used + h4_used < 4:
        # Meaning that less than 4 different hydrophones were used
        return False
    return True
    
def tdoa_and_error(
    parameters : Parameters,
    audio_arrays : list[AudioArray],
    tf_mask_model = None,
    tf_masks_by_tetra: dict[str, np.ndarray] | None = None,
):
    tdoas_measured = []
    tdoas_error_variance = []
    tdoas_mask = []
    for audio_array in audio_arrays:
        external_tf_mask = (
            None
            if tf_masks_by_tetra is None
            else tf_masks_by_tetra.get(audio_array.metadata.tetra_id)
        )
        new_tdoa, new_crb, new_mask, _ = tdoas(
            audio_array,
            use_gcc=False,
            compute_scores=False,
            tf_mask_parameters=parameters.tf_mask_parameters,
            tf_mask_model=tf_mask_model,
            external_tf_mask=external_tf_mask,
        )
        tdoas_measured.append(new_tdoa)
        tdoas_error_variance.append(new_crb)
        tdoas_mask.append(new_mask)

    return tdoas_measured, tdoas_error_variance, tdoas_mask

def fusion(
    parameters : Parameters,
    environment : Environment,
    tdoas_measured,
    tdoas_error_variance,
    tdoas_mask,
    end_tdoa
):
    # Fusion
    if parameters.location_parameters.fusion_type == 'low':
        position_enu, position_error_variance = low_fusion(
            tdoas_measured, tdoas_error_variance, tdoas_mask, environment,
            parameters.location_parameters.projection_plan
        )
    else:
        position_enu, position_error_variance = high_fusion(
            tdoas_measured, tdoas_error_variance, environment,
            projection_plan=parameters.location_parameters.projection_plan
        )

    end_fusion = time()
    if parameters.print_level > 0:
        print(f"▒▒▒▒▒▒▒▒▒▒▒▒ Fusion finished in: {end_fusion - end_tdoa:.2f}s")
    
    return position_enu, position_error_variance

def wave_vector_and_error(
    parameters : Parameters,
    environment : Environment,
    audio_arrays : list[AudioArray],
    fc : float,
    tf_mask_model = None,
    tf_masks_by_tetra: dict[str, np.ndarray] | None = None,
):
    wave_vectors = []
    wave_vectors_error_variance = []
    tetrahedras = [tetra for tetra in environment.tetrahedras.values() if tetra.is_active]
    C = environment.sound_speed
    for tetrahedra, audio_array in zip(tetrahedras, audio_arrays):
        tetrahedral_array = get_tetrahedra(tetrahedra)
        external_tf_mask = (
            None
            if tf_masks_by_tetra is None
            else tf_masks_by_tetra.get(audio_array.metadata.tetra_id)
        )

        u = beamforming_doa(
            parameters.beamforming_parameters,
            fc,
            tetrahedral_array,
            audio_array,
            C,
            tf_mask_parameters=parameters.tf_mask_parameters,
            tf_mask_model=tf_mask_model,
            external_tf_mask=external_tf_mask,
        )

        wave_vectors.append(u.reshape(-1, 1))
        #TODO: implement wave_vectors_error_variance
        wave_vectors_error_variance.append(0)

    return wave_vectors, wave_vectors_error_variance


def _tf_stft_parameters(tf_params):
    from BSS.Utils.associated_dataclasses import StftParameters

    stft = tf_params.stft
    return StftParameters(
        window=stft.window,
        nperseg=stft.n_fft,
        noverlap=stft.n_fft - stft.hop_length,
        nfft=stft.n_fft,
        boundary=stft.boundary,
        padded=stft.padded,
    )


def _network_tf_masks(
    parameters: Parameters,
    audio_arrays: list[AudioArray],
    tf_mask_model=None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    tf_params = parameters.tf_mask_parameters.tf_parameters
    if tf_params is None:
        raise ValueError("TF-mask parameters are required for source separation")

    band_masks_by_tetra = {}
    full_masks_by_tetra = {}
    for audio_array in audio_arrays:
        tetra_id = audio_array.metadata.tetra_id
        band_mask = compute_filtered_tf_mask(
            audio_array,
            parameters.tf_mask_parameters,
            tf_mask_model,
        )
        full_mask = expand_band_mask_to_full_stft(band_mask, tf_params)
        band_masks_by_tetra[tetra_id] = band_mask
        full_masks_by_tetra[tetra_id] = full_mask
    return band_masks_by_tetra, full_masks_by_tetra


def source_tf_masks_for_localization(
    parameters: Parameters,
    environment: Environment,
    audio_arrays: list[AudioArray],
    call_type: str,
    tf_mask_model=None,
) -> tuple[list[tuple[str, dict[str, np.ndarray] | None]], object | None]:
    separation_parameters = getattr(parameters, "source_separation_parameters", None)
    use_neural_mask = (
        parameters.tf_mask_parameters.use_tf_mask
        and parameters.tf_mask_parameters.tf_parameters is not None
        and call_type == "Whistle"
    )

    if not use_neural_mask:
        return [("", None)], None

    if separation_parameters is None or not separation_parameters.enabled:
        return [("", None)], None

    from dataclasses import replace

    from src.source_separation.source_separation_gate import (
        MultiTetraSourceSeparationGate,
        SawadaGateConfig,
        SourceCountGateConfig,
    )

    tf_params = parameters.tf_mask_parameters.tf_parameters
    band_masks_by_tetra, full_masks_by_tetra = _network_tf_masks(
        parameters,
        audio_arrays,
        tf_mask_model,
    )
    stft_parameters = _tf_stft_parameters(tf_params)
    source_count_config = SourceCountGateConfig(
        stft_parameters=stft_parameters,
        method=separation_parameters.source_count_method,
        aggregation=separation_parameters.source_count_aggregation,
        aggregation_quantile=separation_parameters.source_count_aggregation_quantile,
        min_selected_frames=separation_parameters.source_count_min_selected_frames,
        min_active_run_length=separation_parameters.source_count_min_active_run_length,
        min_valid_frequencies=separation_parameters.source_count_min_valid_frequencies,
        min_frequency=tf_params.audio.min_freq,
        max_frequency=tf_params.audio.max_freq,
        mask_mode="all",
    )

    default_sawada_config = SawadaGateConfig()
    sawada_em_parameters = replace(
        default_sawada_config.em_clustering_parameters,
        min_frequency_hz=tf_params.audio.min_freq,
        max_frequency_hz=tf_params.audio.max_freq,
    )
    sawada_config = replace(
        default_sawada_config,
        stft_parameters=stft_parameters,
        min_sources_for_separation=(
            separation_parameters.min_sources_for_separation
        ),
        min_reliable_tetrahedra=separation_parameters.min_reliable_tetrahedra,
        global_source_strategy=separation_parameters.global_source_strategy,
        global_source_quantile=separation_parameters.global_source_quantile,
        require_all_tetrahedra_separated=(
            separation_parameters.require_all_tetrahedra_separated
        ),
        align_sources_across_tetrahedra=(
            separation_parameters.align_sources_across_tetrahedra
        ),
        em_clustering_parameters=sawada_em_parameters,
    )
    gate = MultiTetraSourceSeparationGate(source_count_config, sawada_config)
    decision = gate.process(
        audio_arrays,
        environment,
        active_tf_masks_by_tetra=full_masks_by_tetra,
    )

    if parameters.print_level > 0:
        print(
            "▒▒▒▒▒▒▒▒▒▒▒▒ Source separation: "
            f"k={decision.global_n_sources}, "
            f"separate={decision.should_separate}, reason={decision.reason}"
        )
        if parameters.print_level > 1:
            for count in decision.source_counts:
                print(
                    f"  {count.tetra_id}: k={count.estimated_n_sources}, "
                    f"reliable={count.reliable}, "
                    f"valid_freq={count.valid_frequency_count}, "
                    f"active_bins={count.active_bin_ratio:.1%}"
                )

    if not decision.should_separate or not decision.source_masks_by_source:
        return [("", band_masks_by_tetra)], decision

    source_masks = []
    for source_index, masks_by_tetra in enumerate(decision.source_masks_by_source):
        combined_masks = {}
        for audio_array in audio_arrays:
            tetra_id = audio_array.metadata.tetra_id
            if tetra_id not in masks_by_tetra or tetra_id not in band_masks_by_tetra:
                continue
            sawada_band_mask = extract_band_mask_from_full_stft(
                masks_by_tetra[tetra_id],
                tf_params,
            )
            combined_masks[tetra_id] = (
                np.asarray(band_masks_by_tetra[tetra_id], dtype=bool)
                & np.asarray(sawada_band_mask, dtype=bool)
            )
        if combined_masks:
            source_masks.append((f"S{source_index + 1}", combined_masks))

    if not source_masks:
        return [("", band_masks_by_tetra)], decision
    return source_masks, decision


def localize_audio_group(
    parameters: Parameters,
    environment: Environment,
    audio_arrays: list[AudioArray],
    call_type: str,
    event_start_dt,
    duration: float,
    tf_mask_model=None,
    tf_masks_by_tetra: dict[str, np.ndarray] | None = None,
    timing_reference: float | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if not audio_arrays:
        return None, None, "reject_empty_source_group"

    from dataclasses import replace

    audio_arrays = [
        AudioArray(
            replace(audio_array.metadata),
            environment.tetrahedras[audio_array.metadata.tetra_id],
            audio_array.use_h4,
            data_array=audio_array.data_array.copy(),
        )
        for audio_array in audio_arrays
    ]

    central_frequency = audio_arrays[0].metadata.central_frequency
    if central_frequency is None:
        central_frequency, _, _ = signal_characteristics(audio_arrays)

    use_bf = parameters.beamforming_parameters.use_bf and call_type == "Whistle"
    if use_bf:
        try:
            wave_vectors, wave_vectors_error_variance = wave_vector_and_error(
                parameters,
                environment,
                audio_arrays,
                central_frequency,
                tf_mask_model,
                tf_masks_by_tetra,
            )
        except ValueError as exc:
            print(f"Warning : Beamforming mask is not usable ({exc})")
            return None, None, "reject_beamforming_mask"
        end_bf = time()
        if parameters.print_level > 0 and timing_reference is not None:
            print(
                "▒▒▒▒▒▒▒▒▒▒▒▒ wave vectors computed in: "
                f"{end_bf - timing_reference:.2f}s"
            )

        position_enu, position_error_variance = fusion_bf(
            wave_vectors,
            wave_vectors_error_variance,
            environment,
            parameters.location_parameters.projection_plan,
        )
    else:
        tdoas_measured, tdoas_error_variance, tdoas_mask = tdoa_and_error(
            parameters,
            audio_arrays,
            tf_mask_model,
            tf_masks_by_tetra,
        )

        if not tdoas_mask_check(tdoas_mask):
            print("Warning : Tdoas are not usable")
            return None, None, "reject_tdoa"

        end_tdoa = time()
        if parameters.print_level > 0 and timing_reference is not None:
            print(f"▒▒▒▒▒▒▒▒▒▒▒▒ TDOAs computed in: {end_tdoa - timing_reference:.2f}s")

        position_enu, position_error_variance = fusion(
            parameters,
            environment,
            tdoas_measured,
            tdoas_error_variance,
            tdoas_mask,
            end_tdoa,
        )

    if position_enu is None:
        return None, None, "reject_fusion"

    return position_enu, position_error_variance, "ok"


def source_status_summary(statuses: list[str]) -> str:
    if not statuses:
        return "reject_no_source_group"
    if all(status == "ok" for status in statuses):
        return "ok"
    if any(status == "ok" for status in statuses):
        return "partial_ok:" + ",".join(statuses)
    return statuses[0] if len(statuses) == 1 else "reject_all:" + ",".join(statuses)


###############################
########## Full loop ##########
###############################

def one_iteration(parameters: Parameters, audio_files: list[str], beluga_sounds: list[pd.Series],
                 call_type: str, offset: timedelta, environment: Environment, sound_mask, tf_mask_model = None):

    start_detection = time()

    try:
        audio_arrays, duration, event_start_dt = setup_to_detection(
            parameters, audio_files, beluga_sounds, call_type, offset, environment, sound_mask
        )
        if audio_arrays is None:
            print("WARNING : Audio arrays is None, indicating a setup to detection issue")
            return [], [], [], [], [], None, None, "reject_setup"
    except Exception as e:
        print(f"▒▒▒▒▒▒▒▒▒▒▒▒ Pas de béluga ou erreur: {e}")
        return [], [], [], [], [], None, None, "reject_setup"

    end_detection = time()
    if parameters.print_level > 0:
        print(f"▒▒▒▒▒▒▒▒▒▒▒▒ Detection made in: {end_detection - start_detection:.2f}s")

    # Pre-filtering from audio_arrays with frequency range
    for i in range(len(audio_arrays)):
        if audio_arrays[i].data_array.shape[1] == 0:
            print("WARNING : Canals matching went wrong, audio array's shape is empty")
            return [], [], [], [], [], None, None, "reject_setup"
        audio_arrays[i] = filter_audio_array_from_calltype(audio_arrays[i], parameters.pre_filter_parameters)

    # VMD filtering
    if parameters.vmd_denoise_parameters.use_vmd:
        for i in range(len(audio_arrays)):
            audio_arrays[i] = vmd_denoise(audio_arrays[i], parameters.vmd_denoise_parameters)

    end_denoising = time()
    if parameters.print_level > 0:
        print(f"▒▒▒▒▒▒▒▒▒▒▒▒ Denoising made in: {end_denoising - end_detection:.2f}s")

    central_frequency, frequency_range, snrs_list = signal_characteristics(audio_arrays)
    for i, snrs in enumerate(snrs_list):
        audio_arrays[i].metadata.snr_power = snrs
        audio_arrays[i].update_snr(snrs)
        audio_arrays[i].metadata.central_frequency = central_frequency
        audio_arrays[i].metadata.frequency_range = frequency_range

    if parameters.wt_denoise_parameters.use_wt:
        audio_arrays = wt_denoise(audio_arrays, parameters.wt_denoise_parameters)

    # Second filtering
    for i in range(len(audio_arrays)):
        audio_arrays[i] = filter_audio_array(audio_arrays[i], parameters.pre_filter_parameters)

    source_mask_groups, _separation_decision = source_tf_masks_for_localization(
        parameters,
        environment,
        audio_arrays,
        call_type,
        tf_mask_model,
    )

    positions_enu = []
    positions_error_variance = []
    associated_times = []
    durations = []
    source_labels = []
    statuses = []
    n_groups = len(source_mask_groups)

    for source_index, (source_label, tf_masks_by_tetra) in enumerate(source_mask_groups):
        if parameters.print_level > 0 and n_groups > 1:
            print(f"▒▒▒▒▒▒▒▒▒▒▒▒ Localisation source {source_index + 1}/{n_groups}")

        position_enu, position_error_variance, status = localize_audio_group(
            parameters,
            environment,
            audio_arrays,
            call_type,
            event_start_dt,
            duration,
            tf_mask_model,
            tf_masks_by_tetra,
            timing_reference=end_denoising,
        )
        statuses.append(status)
        if position_enu is None:
            continue

        positions_enu.append(position_enu)
        positions_error_variance.append(position_error_variance)
        associated_times.append(event_start_dt)
        durations.append(duration)
        source_labels.append(source_label)

    status = source_status_summary(statuses)
    return (
        positions_enu,
        positions_error_variance,
        associated_times,
        durations,
        source_labels,
        event_start_dt,
        duration,
        status,
    )


def positions_from_audio(model_path :str, env_path:str, param_path:str, audio_files:list[str]) -> tuple[list[np.ndarray], list[np.ndarray], list, list[float], list[str],list, list[float], list[str], list[str], list[pd.DataFrame]] :

    """Main loop to output positions from an audio.

    Args:
        model_path (str): Path to the detection model
        env_path (str): Path to environment json
        param_path (str): Path to parameters json
        audio_files (list[str]): List of size the number of tetrahedras in
        the environment with path to hand-synchronized 4-channels wavs

    Returns:
    
        positions_enu (list[np.ndarray]) : Positions in the ENU referential
        positions_error_variance (list[np.ndarray]) : Expected variance of errors
        associated_times (list[float]) : Beginning times of beluga signals
        durations (list[float]) : Durations of beluga signals
        call_types(list[str]) : List of call types matching the used sounds
    """
    ##### JSON reading #####
    #model = MultiHeadResNet(num_call_types=4, pretrained=True)
    model = MobileNetMultilabel(num_classes=4, pretrained=True)
    model.load_model(model_path)
    parameters = Parameters(param_path)
    environment = Environment(env_path, parameters.location_parameters.use_h4)

    ##### Time frequency mask model #####
    tf_mask_model = None
    if parameters.tf_mask_parameters.use_tf_mask:
        tf_mask_model = load_tf_mask_model(parameters.tf_mask_parameters.tf_parameters, parameters.tf_mask_parameters.use_phase_aware_network)
    
    ##### Time variables initialisation #####
    iters = 0
    main_start = time()
    offset = timedelta()
    
    ##### Lists of outputs #####
    event_times = []
    event_durations = []
    event_call_types = []
    event_status = []
    positions_enu = []
    positions_error_variance = []
    associated_times = []
    durations = []
    results_dfs = []
    call_types = []
    all_sounds = ["Whistle", "HFPC", "ECHO", "CC", "Noise"]
    masks_dict = {sound: [] for sound in all_sounds}

    ##### Initialize detection module #####
    max_size = None
    #csv_path = r".\test_data\results\beluga_sounds_debug_qat.csv"
    #if os.path.exists(csv_path):
    #    os.remove(csv_path)
    for audio_file in audio_files:
        path_ref_tetra = os.path.basename(audio_file)
        audio_start_time = get_audio_start_time(path_ref_tetra) + offset
        spect_generator = SpectrogramGenerator(
            n_fft=2048,
            hop_length=200,
            n_mels=64,
            fmin=200,
            sample_rate=192000, 
        )

        long_audio, sample_rate, _ = spect_generator.load_audio(audio_file)

        if parameters.print_level >0:
            print(f"▒▒▒▒▒▒▒▒▒▒▒▒ {audio_start_time}")
        #donne le dataframe avec les resultats pour chaque 
        results_df = run_pipeline_overlaps_long_spects(
            long_audio,
            audio_start_time,
            sample_rate,
            model,
            spect_generator=spect_generator,
            debug=False,
            batch_size=64,
            call_model_window_s=1,
            hydrophone_sensitivity=environment.sensitivity,
            seconds_to_process=None,
            skip_first_n_seconds=0
        )
        #tetra_tag = os.path.basename(audio_file).split(".")[0]
        #out_dir = r".\test_data\results"
        #os.makedirs(out_dir, exist_ok=True)
        #per_tetra_csv = os.path.join(out_dir, f"detections_{tetra_tag}.csv")
        #if "Call_Detection" in results_df.columns:
        #    detections_only = results_df[results_df["Call_Detection"] > 0.5].copy()
        #else:
        #    detections_only = results_df.copy()
        #detections_only.to_csv(per_tetra_csv, index=False)

        #if parameters.print_level > 0:
        #    print(f"[DEBUG] wrote {len(detections_only)} detections to {per_tetra_csv}")

        #if parameters.print_level > 0:
        #    print(f"[DEBUG] wrote per-tetra detections: {per_tetra_csv}  ({len(results_df)} rows)")
        if max_size is None:
            max_size = len(results_df)
        else:
            max_size = min(max_size, len(results_df))
        # Noises are indexes where no beluga call was identified
        masks_dict["Noise"].append(((~results_df["Whistle"]) & (~results_df["HFPC"]) & (~results_df["ECHO"])&(~results_df["CC"])).values)
        for sound in all_sounds[:4]:
            masks_dict[sound].append(results_df[sound].values)
        results_dfs.append(results_df)
    results_dfs = [result_df[:max_size] for result_df in results_dfs] 

    iterable_dict = {sound : (np.array([mask[:max_size] for mask in masks_dict[sound]]).T) for sound in all_sounds}
    
    iters = 0
    
    ##### Beginning of the loop #####
    for whistle_mask, hfpc_mask, echo_mask, cc_mask, noise_mask in zip(iterable_dict['Whistle'], iterable_dict['HFPC'], iterable_dict['ECHO'], iterable_dict['CC'], iterable_dict['Noise']):
        if parameters.print_level > 1:
            print(f"Itération : {iters}")
        try :        
            for sound_mask, call_type in zip([whistle_mask, hfpc_mask, echo_mask, cc_mask],["Whistle","HFPC","ECHO","CC"]):
                if np.sum(sound_mask) >= 2:
                    if call_type in ['Whistle','HFPC']:
                        # TODO AJOUTER QQC POUR CHOISIR LES CANAUX A PARSER EN FONCTION DU MASQUE? PEUT ETRE UN SOUS ENVIRONNEMENT ?
                        sounds_lines = [result_df.loc[iters] for result_df in results_dfs]
                        if parameters.print_level >1:
                            print(f'offset : {offset}')
                        (
                            event_positions_enu,
                            event_positions_error_variance,
                            event_associated_times,
                            event_position_durations,
                            event_source_labels,
                            event_time,
                            event_duration,
                            status,
                        ) = one_iteration(
                            parameters,
                            audio_files,
                            sounds_lines,
                            call_type,
                            offset,
                            environment,
                            sound_mask,
                            tf_mask_model,
                        )

                        if event_time is not None:
                            event_times.append(event_time)
                            event_durations.append(event_duration)
                            event_call_types.append(call_type)
                            event_status.append(status)         

                        for (
                            position_enu,
                            position_error_variance,
                            associated_time,
                            duration,
                            source_label,
                        ) in zip(
                            event_positions_enu,
                            event_positions_error_variance,
                            event_associated_times,
                            event_position_durations,
                            event_source_labels,
                        ):
                            positions_enu.append(position_enu)
                            positions_error_variance.append(position_error_variance)
                            associated_times.append(associated_time)
                            durations.append(duration)
                            call_types.append(
                                call_type if not source_label else f"{call_type}_{source_label}"
                            )
                    
        except Exception as e:
            
            print(f"▒▒▒▒▒▒▒▒▒▒▒▒ Attention erreur: {e}")
        
        if parameters.max_position_frames is not None and iters > parameters.max_position_frames:
            main_end = time()
            print(f'Reached max iters in {main_end- main_start}s')
    
            return (
                positions_enu, positions_error_variance, associated_times, durations, call_types,
                event_times, event_durations, event_call_types, event_status,
                results_dfs,   # 👈 NEW
            )

        
        iters += 1
        offset += timedelta(seconds=1)

    main_end = time()
    
    print(f'Finished the full pipeline in {main_end- main_start}s')
    
    return (
        positions_enu, positions_error_variance, associated_times, durations, call_types,
        event_times, event_durations, event_call_types, event_status,
        results_dfs,   # 👈 NEW
    )
