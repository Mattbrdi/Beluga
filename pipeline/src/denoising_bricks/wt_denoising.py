import sys
from pathlib import Path


#TODO: Fix this by removing the need for sys and pathlib
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.sub_classes import Environment, Parameters, AudioMetadata, AudioArray, WTDenoiseParameters
from wavelet_denoising.denoising import WaveletDenoising
from wavelet_denoising.src.si_acf.si_acf import level_determination
from src.utils.color_prints import LOG_STYLES, cprint

from copy import deepcopy

def wt_brick(audio_arrays : list[AudioArray], parameters: WTDenoiseParameters):
    """Perform Wavelet decomposition and filter components based on the thresholding method used.

    Args:
        audio_array (AudioArray): Class containing four channel audio data.
        parameters (WTDenoiseParameters): Parameters for denoising.

    Returns:
        tuple: denoised_arrays, noisy_array
    """
    
    # determine the decomposition level
    level = 1

    # Create the WaveletDenoising class which runs the full decomposition - thresholding - reconstruction pipeline
    wd = WaveletDenoising(normalize=False,
                          wavelet=parameters.wavelet,
                          transform=parameters.transform,
                          level=1,
                          thr_mode=parameters.thr_mode,
                          recon_mode=parameters.recon_mode,
                          selected_level=level,
                          method=parameters.method,
                          energy_perc=parameters.energy_perc
                          )
    
    



    # Deep copy audio_arrays so we can return both denoised and original arrays
    denoised_arrays = deepcopy(audio_arrays)
    for i in range(len(denoised_arrays)):
        #TODO: level determination should be done by specifing the sampling and fundamental frequency not only for si_acf
        # level determination for denoising for si_acf method but also other methods 
        wd.sampling_frequency = denoised_arrays[i].metadata.sample_rate
        wd.fundamental_frequency = denoised_arrays[i].metadata.central_frequency


        level = level_determination(wd.fundamental_frequency, wd.sampling_frequency)

        wd.level = level 
        wd.nlevel = level
        wd.selected_level = level

        for j in range(len(denoised_arrays[i].data_array)):
            # Denoise hydrophone j for tetra i 
            array_to_process = denoised_arrays[i].data_array[j]
            denoised_arrays[i].data_array[j] = wd.fit(array_to_process)   
  
    return denoised_arrays

def wt_denoise(audio_arrays : list[AudioArray], parameters : WTDenoiseParameters):
    """Denoising function using the WT thresholding method.

    Args:
        audio_array (AudioArray): An AudioArray corresponding to one tetrahedra. 
        parameters (WTDenoiseParameters): Parameters ruling the denoising.
    
    Returns:
        audio_array(AudioArray) : The filtered audio_array
    """
    if not parameters.use_wt:
        return audio_arrays
    
    # sample_rate =  audio_arrays[0].metadata.sample_rate
    # #TODO: redundent sample and fundamnetal frequency input
    # parameters.fs = sample_rate
    # if audio_arrays[0].metadata.central_frequency is not None:
    #     parameters.ff = audio_arrays[0].metadata.central_frequency
    # else:
    #     parameters.ff = 2000

    denoised_arrays = wt_brick(audio_arrays, parameters)
    return denoised_arrays


    