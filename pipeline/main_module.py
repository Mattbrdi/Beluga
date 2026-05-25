"""Main function modules to go from wav to positions."""

import os
from datetime import timedelta
from time import time
import numpy as np
import soundfile as sf
import pandas as pd
from src.detection_bricks.mono_audio_detection import SpectrogramGenerator, MobileNetMultilabel, get_audio_start_time, run_pipeline_overlaps_long_spects
from src.detection_bricks.canals_matching import spotting_to_location_preparation
from src.utils.sub_classes import Environment, Parameters, AudioMetadata, AudioArray
from src.location_bricks.frequencies_filtering import filter_audio_array, filter_audio_array_from_calltype
from src.denoising_bricks.vmd_denoising import vmd_denoise
from src.location_bricks.tdoa_brick import tdoas
from src.location_bricks.low_level_fusion import low_fusion
from src.location_bricks.high_level_fusion import high_fusion
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
        # if event_duration <= 0: #mb55 modif, j'ai mis en commentaire
        #     return None, None, None
    
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
    

###############################
########## Full loop ##########
###############################

def one_iteration(parameters: Parameters, audio_files: list[str], beluga_sounds: list[pd.Series],
                 call_type: str, offset: timedelta, environment: Environment, sound_mask):

    start_detection = time()

    try:
        audio_arrays, duration, event_start_dt = setup_to_detection(
            parameters, audio_files, beluga_sounds, call_type, offset, environment, sound_mask
        )
        if audio_arrays is None:
            print("WARNING : Audio arrays is None, indicating a setup to detection issue")
            return None, None, None, None, "reject_setup"
    except Exception as e:
        print(f"▒▒▒▒▒▒▒▒▒▒▒▒ Pas de béluga ou erreur: {e}")
        return None, None, None, None, "reject_setup"

    end_detection = time()
    if parameters.print_level > 0:
        print(f"▒▒▒▒▒▒▒▒▒▒▒▒ Detection made in: {end_detection - start_detection:.2f}s")

    # Pre-filtering from audio_arrays with frequency range
    for i in range(len(audio_arrays)):
        if audio_arrays[i].data_array.shape[1] == 0:
            print("WARNING : Canals matching went wrong, audio array's shape is empty")
            return None, None, None, None, "reject_setup"
        audio_arrays[i] = filter_audio_array_from_calltype(audio_arrays[i], parameters.pre_filter_parameters)

    # VMD filtering
    if parameters.vmd_denoise_parameters.use_vmd:
        for i in range(len(audio_arrays)):
            audio_arrays[i] = vmd_denoise(audio_arrays[i], parameters.vmd_denoise_parameters)

    end_denoising = time()
    if parameters.print_level > 0:
        print(f"▒▒▒▒▒▒▒▒▒▒▒▒ Denoising made in: {end_denoising - end_detection:.2f}s")
    ######
    ######
    #####
    # (mb55) ZONE OU PLACER LA DETECTION DU NOMBRE DE SOURCE ET LA SEPARATION DE SOURCES 
    #ici c'est les audio pour une seconde de temps, il faut traiter audio_arrays qui contient les différend tétraèdre
    #######
    ######
    ########
    
    # Signal characteristics
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

    # TDOAs and CRBs
    tdoas_measured = []
    tdoas_error_variance = []
    tdoas_mask = []
    for audio_array in audio_arrays:
        new_tdoa, new_crb, new_mask, _ = tdoas(audio_array, use_gcc=False, compute_scores=False)
        tdoas_measured.append(new_tdoa)
        tdoas_error_variance.append(new_crb)
        tdoas_mask.append(new_mask)

    associated_time = event_start_dt

    if not tdoas_mask_check(tdoas_mask):
        print("Warning : Tdoas are not usable")
        return None, None, associated_time, duration, "reject_tdoa"

    end_tdoa = time()
    if parameters.print_level > 0:
        print(f"▒▒▒▒▒▒▒▒▒▒▒▒ TDOAs computed in: {end_tdoa - end_denoising:.2f}s")

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

    if position_enu is None:
        return None, None, associated_time, duration, "reject_fusion"

    return position_enu, position_error_variance, associated_time, duration, "ok"


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
                        position_enu, position_error_variance, associated_time, duration, status = one_iteration(parameters, audio_files, sounds_lines, call_type, offset, environment, sound_mask)

                        if associated_time is not None:
                            event_times.append(associated_time)
                            event_durations.append(duration)
                            event_call_types.append(call_type)
                            event_status.append(status)         

                        if position_enu is not None:
                            positions_enu.append(position_enu)
                            positions_error_variance.append(position_error_variance)
                            associated_times.append(associated_time)
                            durations.append(duration)
                            call_types.append(call_type)
                    
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
