## THe objecgive of this file is to be able to test diffenrt denoising techniques to increase SNR
## Only SNR objectives here so we only comptes SNR from audio files that the only thing we wanna do 

import sys
from pathlib import Path

# Allow this test file to be run directly with `python src/tests/denoising_benchmark.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import copy 

from src.detection_bricks.mono_audio_detection import SpectrogramGenerator, MobileNetMultilabel, get_audio_start_time, run_pipeline_overlaps_long_spects
import os
from time import time
from datetime import timedelta
from src.utils.sub_classes import Environment, Parameters, AudioMetadata, AudioArray
from src.location_bricks.frequencies_filtering import filter_audio_array, filter_audio_array_from_calltype
from src.denoising_bricks.vmd_denoising import vmd_denoise
from main_module import signal_characteristics, setup_to_detection

audio_path =  [r"C:\Users\amine\Desktop\Canada\transfer_amine\test_data\full audios\8296\8296.240729065600.wav", 
               r"C:\Users\amine\Desktop\Canada\transfer_amine\test_data\full audios\8295\8295.240729065600.wav",
               ]    
model_path = 'jsons/models/mobile_net_8_layers_qat.pt'
param_path = 'jsons/parameters/default_parameters.json'
env_path = 'jsons/environments/env_cacouna.json'

def butterworth_filtering(audio_arrays : list[AudioArray], parameters : Parameters):
    for i in range(len(audio_arrays)):
        if audio_arrays[i].data_array.shape[1] == 0:
            print("WARNING : Canals matching went wrong, audio array's shape is empty")
            return None, None, None, None, "reject_setup"
        audio_arrays[i] = filter_audio_array_from_calltype(audio_arrays[i], parameters.pre_filter_parameters)

    # VMD filtering
    if parameters.vmd_denoise_parameters.use_vmd:
        for i in range(len(audio_arrays)):
            audio_arrays[i] = vmd_denoise(audio_arrays[i], parameters.vmd_denoise_parameters)

def filter(audio_arrays : list[AudioArray]):
    pass

def compute_SNR(audio_arrays : list[AudioArray]):
    central_frequency, frequency_range, snrs_list = signal_characteristics(audio_arrays)
    return snrs_list


def denoising_iteration(parameters: Parameters, audio_files: list[str], beluga_sounds: list[pd.Series],
                 call_type: str, offset: timedelta, environment: Environment, sound_mask):
    start_detection = time()
    try:
        audio_arrays, duration, event_start_dt = setup_to_detection(
            parameters, audio_files, beluga_sounds, call_type, offset, environment, sound_mask
        )
        if audio_arrays is None:
            print("WARNING : Audio arrays is None, indicating a setup to detection issue")
            return None
    except Exception as e:
        print(f"▒▒▒▒▒▒▒▒▒▒▒▒ Pas de béluga ou erreur: {e}")
        return None

    end_detection = time()
    if parameters.print_level > 0:
        print(f"▒▒▒▒▒▒▒▒▒▒▒▒ Detection made in: {end_detection - start_detection:.2f}s")

    butterworth_arrays = copy.deepcopy(audio_arrays)
    
    filtered_arrays = copy.deepcopy(audio_arrays)
    
    butterworth_filtering(butterworth_arrays, parameters)

    butterworth_filtering(filtered_arrays, parameters)

    snrs_original = compute_SNR(audio_arrays) # looks like (2, 4 ) array

    snrs_butterworth = compute_SNR(butterworth_arrays)

    snrs_filtered = compute_SNR(filtered_arrays)

    all_snrs = dict()
    all_snrs["original"] = snrs_original
    all_snrs["butterworth"] = snrs_butterworth
    all_snrs["filtered"] = snrs_filtered
    
    return all_snrs

def plot_snrs(valid_iters, all_snrs):
    # all snrs are in the format
    # dict with two keys : tetra1 and tetra2, each tetra has 3 keys : original, butterworth, filetred, and each one of these has 4 snr values
    tetras = ["tetra1", "tetra2"]
    labels = ["original", "butterworth", "filtered"]
    
    all_snr_list = [[], []]

    for i, tetra in enumerate(tetras):
        all_snr_list[i] = all_snrs[tetra] # dict of keys labels and 4 channels each
        all_snr_list_i_t = np.array([all_snr_list[i][label] for label in labels]) # list (3, 4)
        all_snr_list_i = all_snr_list_i_t.transpose(2,0,1) # list (4, 3)
        all_snr_list[i] = all_snr_list_i
           
     # snrs_original_t = np.transpose(all_snrs["original"])
    # snrs_butterworth_t = np.transpose(all_snrs["butterworth"])
    # snrs_filtered_t = np.transpose(all_snrs["filtered"])

    # snrs_hydro_0 = np.array([snrs_original_t[0], snrs_butterworth_t[0], snrs_filtered_t[0]])
    # snrs_hydro_1 = np.array([snrs_original_t[1], snrs_butterworth_t[1], snrs_filtered_t[1]])
    # snrs_hydro_2 = np.array([snrs_original_t[2], snrs_butterworth_t[2], snrs_filtered_t[2]])
    # snrs_hydro_3 = np.array([snrs_original_t[3], snrs_butterworth_t[3], snrs_filtered_t[3]])

    N = len(all_snr_list[0][0][0])
    x = np.arange(N)

    # hydro_data = [
    #     snrs_hydro_0,
    #     snrs_hydro_1,
    #     snrs_hydro_2,
    #     snrs_hydro_3,
    # ]


    fig, axes = plt.subplots(2, 4, figsize=(12, 8), sharex=True, sharey=True)

    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            for k, snr in enumerate(all_snr_list[i][j]):
                print(f"tetra {i} - Hydro {j} - signal {labels[k]} - {snr[-1]}")
                ax.scatter(valid_iters, snr, label=labels[k])
            ax.set_yscale("log")

            ax.set_title(f"tetra {i} - Hydro {j}" )
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("SNR")
            ax.grid(True)
            ax.legend()

    plt.tight_layout()
    plt.show()


def denoising_benchmark(model_path :str, env_path:str, param_path:str, audio_files:list[str]):
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
    
    valid_iters = []
    snrs_tot = dict()
    snrs_tot["tetra1"] = dict()
    snrs_tot["tetra2"] = dict()

    snrs_tot["tetra1"]["original"] = []
    snrs_tot["tetra1"]["butterworth"] = []
    snrs_tot["tetra1"]["filtered"] = []

    snrs_tot["tetra2"]["original"] = []
    snrs_tot["tetra2"]["butterworth"] = []
    snrs_tot["tetra2"]["filtered"] = []

    ##### Beginning of the loop #####
    for whistle_mask, hfpc_mask, echo_mask, cc_mask, noise_mask in zip(iterable_dict['Whistle'], iterable_dict['HFPC'], iterable_dict['ECHO'], iterable_dict['CC'], iterable_dict['Noise']):
        if parameters.print_level > 1:
            print(f"Itération : {iters}")
        try :        
            for sound_mask, call_type in zip([whistle_mask, hfpc_mask, echo_mask, cc_mask],["Whistle","HFPC","ECHO","CC"]):
                if np.sum(sound_mask) >= 2:
                    if call_type in ['Whistle','HFPC']:
                        sounds_lines = [result_df.loc[iters] for result_df in results_dfs]
                        if parameters.print_level >1:
                            print(f'offset : {offset}')
                        snrs = denoising_iteration(parameters, audio_files, sounds_lines, call_type, offset, environment, sound_mask)
                        
                        if snrs is not None:
                            snrs_tot["tetra1"]["original"].append(snrs["original"][0])
                            snrs_tot["tetra1"]["butterworth"].append(snrs["butterworth"][0])
                            snrs_tot["tetra1"]["filtered"].append(snrs["filtered"][0])

                            snrs_tot["tetra2"]["original"].append(snrs["original"][1])
                            snrs_tot["tetra2"]["butterworth"].append(snrs["butterworth"][1])
                            snrs_tot["tetra2"]["filtered"].append(snrs["filtered"][1])
                            valid_iters.append(iters)

        except Exception as e:
            
            print(f"▒▒▒▒▒▒▒▒▒▒▒▒ Attention erreur: {e}")
        
        if parameters.max_position_frames is not None and iters > parameters.max_position_frames:
            main_end = time()
            print(f'Reached max iters in {main_end- main_start}s')
            if snrs is not None:
                plot_snrs(valid_iters, snrs_tot)    
            return None
        
        iters += 1
        offset += timedelta(seconds=1)
    
    if snrs is not None:
        plot_snrs(valid_iters, snrs_tot)

    main_end = time()
    
    print(f'Finished the full pipeline in {main_end- main_start}s')
    
    return None

if __name__ == "__main__":
    denoising_benchmark(model_path, env_path, param_path, audio_path)
