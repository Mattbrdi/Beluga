"""Functions to perform some performance evaluation."""

import pandas as pd
import json
import os
import numpy as np
from scipy.signal import correlate
from time import time
from typing import Optional
import glob

from src.utils.sub_classes import Parameters, AudioArray, Parameters, Environment
from src.denoising_bricks.vmd_denoising import vmd_denoise
from src.location_bricks.frequencies_filtering import filter_audio_array
from src.location_bricks.tdoa_brick import tdoas
from src.location_bricks.low_level_fusion import low_fusion
from src.location_bricks.high_level_fusion import high_fusion
from src.utils.wav_parser import wav_to_array, output_wav_file


##################################
##### Intermediary functions #####
##################################

def get_audios(audio_folder_path: str)-> list[list[str]]:
    """Finds all audio pairs from a folder,
    where pairs are named T1_..., T2_...

    Args:
        audio_folder_path (str): Path to the folder containing pairs

    Returns:
        all_audios (list[list[str]]): Big list of size folder size / 2,
        containing small lists of size the number of tetrahedas
    """
    audio_files = sorted(glob.glob(os.path.join(audio_folder_path, "T*_*.wav")))
    audio_dict = {}
    for f in audio_files:
        base = os.path.basename(f)
        # ID = tout ce qui suit le premier underscore, sans l'extension
        id_part = "_".join(base.split('_')[1:]).replace('.wav', '')
        audio_dict.setdefault(id_part, []).append(f)
        
    all_audios = [sorted(v) for _, v in sorted(audio_dict.items())]
    return all_audios

def get_groundtruths(folder_path : str):
    groundtruth_files = sorted(
        glob.glob(os.path.join(folder_path, "groundtruth_*.json")),
        key=lambda x: os.path.basename(x)
    )
    all_groundtruths = groundtruth_files
    return all_groundtruths

def get_parameters(folder_path : str):
    return sorted(glob.glob(os.path.join('jsons/other_parameters/', "*.json")))

def match_synthetic_audio(wav_path, output_path, environment : Environment, parameters : Parameters, frequency_range = (500,20000)):
    audio_array = wav_to_array(wav_path, True, next(iter(environment.tetrahedras.values())))
    audio_array.metadata.frequency_range = frequency_range
    audio_array = filter_audio_array(audio_array, parameters.pre_filter_parameters)
    six_tdoas, _, _, _ = tdoas(audio_array, parameters.use_gcc, False)
    four_tdoas = np.append(0,six_tdoas[0:3])
    positive_indexes = np.array((np.max(four_tdoas) - four_tdoas)*audio_array.metadata.sample_rate, dtype = int)
    max_idx = np.max(positive_indexes)
    shortened_length = audio_array.data_array.shape[1]- max_idx
    new_array = np.zeros((4,shortened_length))
    for i, tdoa in enumerate(positive_indexes):
        new_array[i, :] = audio_array.data_array[i,tdoa:shortened_length + tdoa]
    output_wav_file(new_array.T, audio_array.metadata.sample_rate, output_path)
    print(f'Successfully matched the four canals and exported them at {output_path}')
    
def one_run(audio_arrays : list[AudioArray], parameters : Parameters, environment : Environment):
    ##### Pre-filtering from audio_arrays with frequency range #####
    for i in range(len(audio_arrays)):
        audio_arrays[i] = filter_audio_array(audio_arrays[i], parameters.pre_filter_parameters)
        
    ##### VMD filtering #####
    if parameters.vmd_denoise_parameters.use_vmd:
        audio_arrays, _ ,_ = vmd_denoise(audio_arrays, parameters.vmd_denoise_parameters)
            
    ##### TDOAs and CRBs computation #####
    tdoas_measured = []
    tdoas_error_variance = []
    tdoas_mask = []
    all_scores = []
    
    for audio_array in audio_arrays:
        ##### Computation #####
        new_tdoa, new_crb, new_mask, scores = tdoas(audio_array, parameters.use_gcc, True)
        
        ##### Agregation #####
        tdoas_measured.append(new_tdoa)
        tdoas_error_variance.append(new_crb)
        tdoas_mask.append(new_mask)
        all_scores.append(scores)
    
    if np.sum(tdoas_mask) < 4:
        print(f"WARNING : Less than three meaningful tdoas, returning NaN values")
        nan_tdoa = np.full((len(tdoas_measured), 6), np.nan)
        nan_variance = np.full((len(tdoas_error_variance), 6), np.nan)
        nan_pos = np.full((3,1), np.nan)
        nan_pos_var = np.full((3,1), np.nan)
        return nan_tdoa, nan_variance, nan_pos, nan_pos_var, all_scores

    ##### Position computation #####
    try:
        if parameters.location_parameters.fusion_type == 'low':
            position_enu, position_error_variance = low_fusion(tdoas_measured, tdoas_error_variance, tdoas_mask, environment, parameters.location_parameters.projection_plan)
        else : #fusion_type == 'high'
            position_enu, position_error_variance = high_fusion(tdoas_measured, tdoas_error_variance, environment, projection_plan = parameters.location_parameters.projection_plan)
    except:
        print(f"WARNING : Fusion failed, returning NaN values")
        nan_tdoa = np.full((len(tdoas_measured), 6), np.nan)
        nan_variance = np.full((len(tdoas_error_variance), 6), np.nan)
        nan_pos = np.full((3,1), np.nan)
        nan_pos_var = np.full((3,1), np.nan)
        return nan_tdoa, nan_variance, nan_pos, nan_pos_var, all_scores
    return tdoas_measured, tdoas_error_variance, position_enu, position_error_variance, all_scores

def reorganised_scores(scores):
    #score_names = ['correlation std', 'num_peaks', 'width_max_peak', 'height_diff', 'height_ratio', 'sharpness']
    correlations = []
    num_peaks = []
    width_max_peaks = []
    height_diffs = []
    height_ratios = []
    sharpnesses = []
    
    for score in scores:
        for subscore in score:
            correlations.append(subscore[0])
            num_peaks.append(subscore[1])
            width_max_peaks.append(subscore[2])
            height_diffs.append(subscore[3])
            height_ratios.append(subscore[4])
            sharpnesses.append(subscore[5])
    return correlations, num_peaks, width_max_peaks, height_diffs, height_ratios, sharpnesses

#########################
##### Main function #####
#########################

def synthetic_benchmark(wav_paths : list[list[str]] , groundtruths_jsons : list[str], parameters_path : list[str], environment : Environment):
    """
    Outputs a dataframe with errors made in tdoas and position
    after a 500-20kHz filtering over synthetic data.

    Args:
        wav_paths (list[list[str]]): All paths to pairs of synthetic wavs 
        groundtruths_jsons (list[str]): All paths to synthetic groundtruths
    """
    if len(wav_paths) != len(groundtruths_jsons):
        raise ValueError("Not the same amount of wavs and groundtruths")
    
    ##### Parsing parameters #####
    parameters_list = [Parameters(param) for param in parameters_path]
    parameters_names = [os.path.basename(param).replace('_parameters.json','').replace('.json','') for param in parameters_path]
    
    ##### Preparing output dataframe #####
    data_columns = ['Audio ID','SNR Power']
    metrics_columns = [name + ' 2D distance error' for name in parameters_names]+ [name + ' 3D distance error' for name in parameters_names] + [name + ' TDOA abs error avg' for name in parameters_names] + [name + ' TDOA error over fech' for name in parameters_names]
    position_columns = ['Exact position'] + [name + ' position' for name in parameters_names]
    tdoas_columns = ['Exact TDOAs'] + [name + ' TDOAs' for name in parameters_names]
    variance_columns = [name + ' expected pos std' for name in parameters_names] + [name + ' expected TDOA std' for name in parameters_names]
    scores_columns = [name + ' ' + score_name for name in parameters_names for score_name in ['correlation std', 'num_peaks', 'width_max_peak', 'height_diff', 'height_ratio', 'sharpness']]
    columns = data_columns + metrics_columns + tdoas_columns + position_columns + variance_columns + scores_columns
    output_dataframe = pd.DataFrame(columns=columns)
    
    i=0
    max_i = len(wav_paths) * len(parameters_list)
    ##### Parsing audio_arrays #####
    for audio_list, groundtruth_json in zip(wav_paths, groundtruths_jsons):
        try:
            ##### Parsing jsons #####
            with open(groundtruth_json, 'r') as file:
                groundtruth = json.load(file)
            all_snrs = groundtruth['SNR power']
            
            ##### Parsing audio_arrays #####
            audio_arrays = [wav_to_array(wav_path,True,tetra,snr) for wav_path, tetra, snr in zip(audio_list, environment.tetrahedras.values(), all_snrs)]
            audio_id = os.path.basename(audio_list[0])[3:].replace('.wav', '') 
            ##### Initialize the output dictionnary #####
            ordered_tdoas = [(-np.array(list(tdoa.values()))).tolist() for tdoa in groundtruth["Whistle tdoas"]]
            exact_position_enu = np.array(groundtruth["Beluga pos ENU"]).astype('float')
            output_results = {
                'Audio ID' : [audio_id],
                'SNR Power' : [all_snrs],
                'Exact TDOAs' : [ordered_tdoas],
                'Exact position' : [exact_position_enu],
            }
            
            ####################################################################
            ##### Modified version of main_module's one iteration function #####
            ####################################################################
            
            for parameters, name in zip(parameters_list, parameters_names):
                beginning_time = time()
                tdoas_measured, tdoas_error_variance, position_enu, position_error_variance, scores = one_run(audio_arrays, parameters, environment)
                output_results[name + ' TDOAs'] = [[tdoa.tolist() for tdoa in tdoas_measured]]
                output_results[name + ' expected TDOA std'] = [[tdoa_err.tolist() for tdoa_err in tdoas_error_variance]]
                output_results[name + ' position'] = [np.ravel(position_enu).tolist()]
                output_results[name + ' expected pos std'] = [np.ravel(position_error_variance).tolist()]
                output_results[name + ' 2D distance error'] = [np.linalg.norm(np.ravel(position_enu)[0:2] - np.ravel(exact_position_enu)[0:2])]
                output_results[name + ' 3D distance error'] = [np.linalg.norm(np.ravel(position_enu) - np.ravel(exact_position_enu))]
                output_results[name + ' TDOA abs error avg'] = [np.mean(np.ravel(tdoas_measured)-np.ravel(ordered_tdoas))]
                output_results[name + ' TDOA error over fech'] = [np.mean(np.abs(np.ravel(tdoas_measured)-np.ravel(ordered_tdoas)))*audio_arrays[0].metadata.sample_rate]
                i+=1
                for score_category, score_name in zip(reorganised_scores(scores),['correlation std', 'num_peaks', 'width_max_peak', 'height_diff', 'height_ratio', 'sharpness']):
                    output_results[name + ' ' + score_name] = np.nanmean(np.array(score_category,dtype=float))
                end_time = time()
                print(f"Progress: {i}/{max_i} - {i/max_i*100:.2f}% - {name}\nTime taken: {end_time-beginning_time:.2f}s - Remaining time: {(end_time-beginning_time)*(max_i-i):.2f}s")
            output_dataframe = pd.concat([output_dataframe, pd.DataFrame(output_results)], ignore_index = True)
        except:
            i+= len(parameters_list)
            print(f"Error processing audio files {audio_list} with groundtruth {groundtruth_json}. Skipping to next pair.")
            continue
    
    return output_dataframe

###################################
##### Analysis of the results #####
###################################

def cross_corr_metrics_analysis(benchmark_path : str, output_path : Optional[str]):
    """
    Analyzes the cross-correlation metrics from a benchmark CSV file and outputs a cleaned CSV file.

    Args:
        benchmark_path (str): Path to the benchmark CSV file.
        output_path (str): Path to save the cleaned CSV file.
    """
    new_benchmark2 = pd.read_csv(benchmark_path)
    
    # Extracting relevant columns
    heights_ratios = new_benchmark2[[col for col in new_benchmark2.columns if 'height_ratio' in col]]
    heights_diffs = new_benchmark2[[col for col in new_benchmark2.columns if 'height_diff' in col]]
    width_peak_df = new_benchmark2[[col for col in new_benchmark2.columns if 'width_max_peak' in col]]
    num_peaks_df = new_benchmark2[[col for col in new_benchmark2.columns if 'num_peaks' in col]]
    
    # Default column names
    default_columns = [col.replace('height_ratio','') for col in new_benchmark2.columns if 'height_ratio' in col]
    new_columns = ['heights_ratios', 'heights_diffs', 'width_peaks', 'num_peaks']

    # Cleaning and renaming columns
    clean_mean_heights = heights_ratios.replace([np.inf, -np.inf], np.nan).mean()
    clean_mean_heights = clean_mean_heights.rename({clean_col: default_col for clean_col, default_col in zip(clean_mean_heights.index, default_columns)})
    
    clean_mean_diffs = heights_diffs.replace([np.inf, -np.inf], np.nan).mean()
    clean_mean_diffs = clean_mean_diffs.rename({clean_col: default_col for clean_col, default_col in zip(clean_mean_diffs.index, default_columns)})
    
    width_peak_means = width_peak_df.replace([np.inf, -np.inf], np.nan).mean()
    width_peak_means=width_peak_means.rename({clean_col: default_col for clean_col, default_col in zip(width_peak_means.index, default_columns)})
    
    num_peaks_means = num_peaks_df.mean()
    num_peaks_means= num_peaks_means.rename({clean_col: default_col for clean_col, default_col in zip(num_peaks_means.index, default_columns)})
    
    final_df = pd.concat([clean_mean_heights, clean_mean_diffs, width_peak_means, num_peaks_means], axis=1, ignore_index=True)
    final_df.columns = new_columns
    
    if output_path is not None:
        final_df.to_csv(output_path)
    
    return final_df

def tdoa_error_analysis(benchmark_path : str):
    benchmark_df = pd.read_csv(benchmark_path)
    tdoa_columns = [col for col in benchmark_df.columns if 'TDOA error over fech' in col]
    tdoa_fech_df = benchmark_df[tdoa_columns]
    return tdoa_fech_df.describe()