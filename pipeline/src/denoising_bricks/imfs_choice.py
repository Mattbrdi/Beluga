import numpy as np
from scipy.signal import correlate
from src.utils.sub_classes import AudioArray, VMDDenoiseParameters
from copy import deepcopy

#########################################
########## Auxiliary functions ##########
#########################################

def order_imfs(imfs, omegas):
    ##### Change the constant terme ####
    order_idx = np.argsort(omegas.ravel())
    return imfs[order_idx, :]

def cc_filter_four_canals(ordered_imfs, audio_array, parameters):
    """Filter the modes via CC for a four canals array."""
    
    processed_imfs = [imfs / np.sqrt(np.sum(imfs**2, axis = 1)).reshape(-1,1) for imfs in ordered_imfs] # IMF of power 1

    if parameters.compute_noisy_imfs:
        noisy_array = deepcopy(audio_array)
    else:
        noisy_array = None
        
    ##### Find the most correlated modes #####
    all_modes_cc = []
    for imf0, imf1 in zip(processed_imfs[0], processed_imfs[1]):
        all_modes_cc.append(np.max(correlate(imf0, imf1, mode='full')))
    if parameters.print_level >=2:
        print(f'Cross correlation scores between the first two canals :\n{all_modes_cc}')
        
    all_modes_cc = np.array(all_modes_cc)
    modes_mask = all_modes_cc > parameters.cc_threshold
    
    ###### Prevent decomposition from going wrong #####
    if np.sum(modes_mask) == 0:
        print("WARNING : No correlated modes, returning original array")
        if parameters.compute_noisy_imfs and noisy_array is not None:
            noisy_array.data_array = np.array([np.sum(imf, axis = 0) for imf in ordered_imfs])
        return audio_array, noisy_array
    
    ##### Otherwise there is at least one mode 
    audio_array.data_array = np.array([np.sum(imf[modes_mask,:], axis = 0) for imf in ordered_imfs])
    if parameters.compute_noisy_imfs and noisy_array is not None:
        noisy_array.data_array = np.array([np.sum(imf[np.logical_not(modes_mask),:], axis = 0) for imf in ordered_imfs])
    return audio_array, noisy_array
    
###################################
########## Main function ##########
###################################

def imf_cc_filter(all_imfs_omegas : list[tuple[np.ndarray,np.ndarray]], parameters : VMDDenoiseParameters, audio_arrays : list[AudioArray]):
    """Filter the modes via CC of all the audio arrays."""
    ##### Order the imfs by frequencies and make them all the same power #####
    ordered_imfs = [order_imfs(imfs_omegas[0], imfs_omegas[1]) for imfs_omegas in all_imfs_omegas]
    
    ##### Prepare for data aggregation #####
    pure_arrays = []
    if parameters.compute_noisy_imfs:
        noisy_arrays = []
    else: 
        noisy_arrays = None
    
    ##### Loop over tetrahedras #####
    for i, audio_array in enumerate(audio_arrays):
        
        #########################
        ##### Main function #####
        #########################
        pure_array, noisy_array = cc_filter_four_canals(ordered_imfs[4*i:4*i+4], audio_array, parameters)
        #########################
        
        pure_arrays.append(pure_array)
        if parameters.compute_noisy_imfs and noisy_arrays is not None:
            noisy_arrays.append(noisy_array)
            
    return pure_arrays, noisy_arrays
        
        
        
    