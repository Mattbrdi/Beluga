"""This module develops the total denoising algorithm, going through all the steps."""

from src.utils import wav_parser
from src.denoising_bricks import woa
from src.denoising_bricks import igff
from src.denoising_bricks import vmd
from src.utils.sub_classes import Parameters, VMDOptions, VMDDenoiseParameters, AudioArray
from src.denoising_bricks.imfs_choice import imf_cc_filter
from joblib import Parallel, delayed

from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
from typing import Optional
import pickle
from time import time
from copy import deepcopy

########################################
########### Mode choice brick ##########
########################################

def filter_imfs(imf, omega, call_type, parameters):
    """Filter imfs corresponding to one canal."""
    ##### IMF filtering #####
    if parameters.remove_dc: # Remove dc if the audio is not prefiltered, useless otherwise
        min_omega_idx = np.argmin(omega)
        imfs_no_dc = np.delete(imf, min_omega_idx, axis = 0) # Remove DC component
        pure_imfs, noisy_imfs, igff_vals = igff.imf_filter(imfs_no_dc, call_type, parameters.imf_threshold)
    else:
        pure_imfs, noisy_imfs, igff_vals = igff.imf_filter(imf, call_type, parameters.imf_threshold)
        ##### Informations #####
        if parameters.print_level>=1:
            print(f'Info : Successfully performed {call_type} VMD')
            if parameters.print_level>=2:
                print(f'Found {np.shape(pure_imfs)[0]} pure IMFs and {np.shape(noisy_imfs)[0]} noisy IMFs over all {np.shape(imf)[0]} IMFs')
                print(f'IGFF values : {igff_vals}')
        
    ##### Agregating #####
    if(noisy_imfs.shape[0]!=0) and parameters.compute_noisy_imfs:
        noisy_data = np.sum(noisy_imfs, axis = 0)
    else:
        noisy_data = None
        
    if(pure_imfs.shape[0]!=0):
        pure_data = np.sum(pure_imfs, axis = 0)
    else:
        raise ValueError('WARNING : No pure imf') # Should never happen tbh
    pure_data = np.sum(pure_imfs, axis = 0)
    
    return pure_data, noisy_data

def igff_brick(imfs_omegas_list, parameters, audio_arrays : list[AudioArray]):
    ##### Initialization #####
    if parameters.compute_noisy_imfs:
        noisy_arrays = deepcopy(audio_arrays)
    else:
        noisy_arrays = None
            
    if len(imfs_omegas_list) != len(audio_arrays)*4:
        raise ValueError("The length of imfs list is not 4* the amount of AudioArray s")

    ##### Computation over all arrays #####
    time0 = time()    
    all_pure_noisy = Parallel(n_jobs=-1, verbose = 50)(
            delayed(filter_imfs)(
                imf,
                omega,
                audio_arrays[i//4].metadata.beluga_call_type,
                parameters
            ) for i, (imf, omega) in enumerate(imfs_omegas_list))
    print(f'IGFF parallélisé took {time()-time0}')
    
    ##### Agregating #####
    for i, (pure_data, noisy_data) in enumerate(all_pure_noisy):
        audio_arrays[i//4].data_array[i-4*i//4] = pure_data
        if parameters.compute_noisy_imfs and noisy_arrays is not None:
            noisy_arrays[i//4].data_array[i-4*i//4] = noisy_data
    
    return audio_arrays, noisy_arrays
    
        
        

##########################################################
########### Intermediary bricks for readability ##########
##########################################################
    

def woa_brick(mono_canal, parameters: VMDDenoiseParameters, fech : int):
    """Apply the WOA algorithm to determine optimal VMD parameters.

    Args:
        mono_canal (np.ndarray): Single channel audio data.
        parameters (VMDDenoiseParameters): Parameters for denoising.
        fech (int): Sampling frequency.

    Returns:
        tuple: Optimal number of modes (K) and penalty factor (alpha).
    """
    K,alpha,pse = woa.woa_algorithm(mono_canal,parameters, fech)
    if parameters.print_level>=1:
        print(f'Successfully performed WOA. K : {K}, alpha : {alpha}, pse : {pse}')
    return K, alpha

##### Prepare parallelisation #####
def canal_to_imfs(mono_canal, sample_rate, parameters):
    imfs , _ , omegas =  vmd.vmd_decomposition(mono_canal, parameters.vmd_options, sample_rate)
    return imfs, omegas


def vmd_brick(audio_arrays : list[AudioArray], parameters: VMDDenoiseParameters):
    """Perform VMD decomposition and filter IMFs based on the call type.

    Args:
        audio_array (AudioArray): Class containing four channel audio data.
        parameters (VMDDenoiseParameters): Parameters for denoising.
        omegas (np.ndarray, optional): Initial guess for center frequencies.

    Returns:
        tuple: audio_array, noisy_array, imfs_list, omegas_list, igff_vals
    """
    
    ###### Initialization #####
    begin_time = time()
    
    all_imfs_omegas = Parallel(n_jobs=-1, verbose = 50)(
        delayed(canal_to_imfs)(
            audio_array.data_array[i],
            audio_array.metadata.sample_rate,
            parameters
        ) for audio_array in audio_arrays for i in range(audio_array.data_array.shape[0]))
    
    print(f'VMD parallélisé took {time()-begin_time}')

    ###### Filtering the modes #####
    if parameters.modes_choice_method == 'igff':
        audio_arrays, noisy_arrays  = igff_brick(all_imfs_omegas, parameters, audio_arrays)
    else: # CC filter
        audio_arrays, noisy_arrays = imf_cc_filter(all_imfs_omegas, parameters, audio_arrays)
    
    ##### Output #####
    end_time = time()
    if parameters.print_level>=1:
        print(f"Finished decomposing all canals in {(end_time-begin_time):.2f}s")
        
    return audio_arrays, noisy_arrays, all_imfs_omegas

##############################################
########### Main denoising function ##########
##############################################
    
def vmd_denoise(audio_arrays : list[AudioArray], parameters : VMDDenoiseParameters):
    """Denoising function using the  VMD algorithm.

    Args:
        audio_array (AudioArray): An AudioArray corresponding to one tetrahedra. 
        parameters (VMDDenoiseParameters): Parameters ruling the denoising.
    
    Returns:
        audio_array(AudioArray) : The filtered audio_array with relevant IMFs
    """
    if parameters.use_woa:
        K, alpha = woa_brick(audio_arrays[0].data_array[0], parameters, audio_arrays[0].metadata.sample_rate)
        parameters.vmd_options.set_k_alpha(K,alpha)
        
    audio_arrays, noisy_arrays, imfs_omegas = vmd_brick(audio_arrays, parameters)
    
    return audio_arrays, noisy_arrays, imfs_omegas