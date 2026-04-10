from src.location_bricks.tdoa_brick import crb_from_pair, snr1_value,snr2_value,snr3_value, transition_bound, tdoas, tdoas_from_h1
from src.location_bricks.frequencies_filtering import filter_data_array
from src.utils.sub_classes import Environment, HydrophonePair, AudioMetadata, AudioArray, Hydrophone, PreFilterParameters
from src.utils.wav_parser import read_wav_file
from src.utils.four_can_generator import generate_gaussian_whistle, generate_correlated_array_from_nparray_int_tdoas
from scipy.optimize import minimize
from main_module import signal_characteristics
from src.utils.plots import plot_spectro
from src.tests.debug_functions import plot_analysis

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, windows
from typing import Optional
import os
import re
from copy import deepcopy
from time import time
import pickle

def mse_cost(y, hat_y):
    return np.mean((y-hat_y)**2)

def rel_mse_cost(y, hat_y):
    return np.mean((np.log(y+1)-np.log(hat_y+1))**2)


def ziv_zakai_value(
    snr : float,
    bandwidth : float,
    duration : float,
    central_frequency : float,
    environment : Environment,
    sample_rate = 384000
    ):
    snr_each_canal = snr + np.sqrt(snr**2 + snr) # SNR = SNR'^2 / (1+ 2 SNR') => SNR' = SNR + \sqrt(SNR^2 + SNR)
    metadata = AudioMetadata('','Whistle', duration, 0.0, 4 * [snr_each_canal], sample_rate, (500,20000), central_frequency)
    hydrophone = Hydrophone('', np.zeros(0),np.zeros(0), snr_each_canal)
    delay_seconds = next(iter(environment.tetrahedras.values())).max_delay_seconds
    hydrophone_pair = HydrophonePair(hydrophone,hydrophone, max_delay_idx=int(delay_seconds*sample_rate))
    crb_value , _ = crb_from_pair(metadata,duration,hydrophone_pair,bandwidth)
    # WARNING : NOT RELEVANT ANYMORE AS THE DISCIRIMINATION BETWEEN FREQUENCIES WAS ADDED TO CRB FROM PAIR
    return crb_value

def load_noises(noises_folder, max_length_idx):
    
    all_noise_arrays = []    
    
    for noise in os.listdir(noises_folder):
        if noise.endswith('.wav'):
            # Charger le fichier audio
            noise_path = os.path.join(noises_folder, noise)
            noise_array , _  = read_wav_file(noise_path)
            noise_array = noise_array[:,:max_length_idx]
            all_noise_arrays.append(noise_array)
    
    return all_noise_arrays

def spectral_power(data_array, sample_rate, frequency_range):
    n_canaux, n_points = np.shape(data_array)
    
    ##### Compute FFT #####
    window = windows.hann(n_points)[None, :]
    signals_win = data_array * window
    spectrum = np.fft.rfft(signals_win, axis=1)
    freqs = np.fft.rfftfreq(n_points, 1.0/sample_rate)
    power = np.abs(spectrum)**2
    
    ##### Band selection #####
    start_idx = np.searchsorted(freqs, frequency_range[0])
    end_idx = np.searchsorted(freqs, frequency_range[1])
    freqs_band = freqs[start_idx:end_idx]
    power_band = power[:, start_idx:end_idx]
    return np.sum(power_band, axis = 1)
    

def parse_one_full_audio(noise_array : np.ndarray, data_arrays : list[np.ndarray], sample_rate, snr):
    
    whistle_arrays = deepcopy(data_arrays)
    
    
    
    #noise_power = np.sum(noise_array**2, axis = 1)
    noise_power = spectral_power(noise_array, sample_rate, (500,20000)) / (20000.-500.)
    #whistle_array_powers = [np.sum(whistle_data_array**2, axis = 1) for whistle_data_array in whistle_arrays]
    whistle_array_powers = [spectral_power(whistle_data_array, sample_rate, (500,20000))/ (900.) for whistle_data_array in whistle_arrays]
    old_maxes = [np.max(np.abs(whistle), axis=1, keepdims = True) for whistle in whistle_arrays]
    
    for i, whistle_power_list in enumerate(whistle_array_powers):
        correctors = snr * noise_power/whistle_power_list
        correctors = correctors.reshape((4,1))
        whistle_arrays[i] *= np.sqrt(correctors)
    
    final_arrays = [whistle_data_array + noise_array for whistle_data_array in whistle_arrays]
        ##### Make sure to stay between [-1,1] #####
    for i in range(len(final_arrays)):
        final_arrays[i] = final_arrays[i] / np.max(np.abs(final_arrays[i])) * old_maxes[i]
    
    ##### Parsing the to AudioArray class #####
    audio_metadatas = [AudioMetadata(tetra_id,'Whistle',1.0 ,0.0, 4 * [snr], sample_rate, (500,20000), 10000) for tetra_id in environment.tetrahedras.keys()]
    audio_arrays = [AudioArray(audio_metadata, tetra_data, True, final_array) for audio_metadata, tetra_data, final_array in zip(audio_metadatas,environment.tetrahedras.values(), final_arrays)]

    return audio_arrays
    

def monte_carlo_error(
    snr : float,
    environment : Environment,
    nb_draws : int,
    list_noises : list[np.ndarray],
    beluga_noise : np.ndarray,
    position : np.ndarray,
    nb_samples_fs : int,
    frame_rate : int = 384000,
    ):
    
    snr_each_canal = snr + np.sqrt(snr**2 + snr)
    all_errors = []
    # Having different pos is mostly useful to assess the 1/fs bound, because at some point the
    # error is constant, just depending on where is the real tau, in other words constant for a
    # pos. So the more pos the better.
    
    for _ in range(nb_samples_fs):
        data_arrays, all_tdoas, _ = generate_correlated_array_from_nparray_int_tdoas(environment, 0.0, position, beluga_noise, frame_rate, 1.0)

        print(f'Expected SNR at each canal : {10*np.log10(snr_each_canal)}')
        
        all_audio_arrays = [parse_one_full_audio(noise, data_arrays, frame_rate, snr_each_canal) for noise in list_noises]
        
        
        all_tests_snrs = []
        all_tests_central_freq = []
        for audio_arrays in all_audio_arrays:
            for audio_array in audio_arrays:
                #plot_analysis(audio_array.data_array, 384000, (500,20000))
                #plot_spectro(audio_array, None, True, -70, 0)
                central_freq , snrs = signal_characteristics(audio_array)
                all_tests_snrs.append(np.mean(snrs))
                all_tests_central_freq.append(central_freq)
        print(f'Estimated central frequency[Hz] : {np.median(all_tests_central_freq)}')
        print(f'Estimated SNR [dB] : {10*np.log10(np.mean(all_tests_snrs))}')
        
        
        for i in range(min(nb_draws, len(all_audio_arrays))):
            for true_tdoa, audio_array in zip(all_tdoas, all_audio_arrays[i]):
                estimated_tdoas  = tdoas_from_h1(audio_array)
                # Le + vient qu'il y a un problème de convention +- des tdoas ...
                error = np.array(estimated_tdoas) + np.array(list(true_tdoa.values())[:len(estimated_tdoas)])

                all_errors.append(error)
    aligned_errors = np.ravel(all_errors)
    error_variance = np.var(aligned_errors)
    return error_variance
            
def ideal_function(
    snr : np.ndarray,
    snr_shift : float,
    omega0 : float,
    omega_bw : float,
    duration : float,
    delay : float,
    sample_rate : float,
):
    """Compute the estimated value of the error with only one snr shift 

    Args:
        snr (np.ndarray): Values of snr (as snr**2/(1+2 snr))
        snr_shift (float): Shift between the two posssible modes of the function
        omega0 (float): Central frequency in Hz . rad
        omega_bw (float): Bandwidth in Hz. rad
        duration (float): 1.0 secondes
        delay (float): 2* max delay entre hydrophones (2* distmax /cw)
        sample_rate (float): 384000Hz

    Returns:
        float: estimated error variance
    """
    values = np.zeros(snr.shape)
    
    shift_indexes = np.where(snr < snr_shift)[0]
    if len(shift_indexes) > 0:
        shift_index = shift_indexes[-1]
    else: 
        shift_index = None
    uniform = delay **2/12.0 #* np.ones(snr[:shift_index].shape)
    barankin_bound = 12.0 * np.pi /(omega_bw**3 * duration * snr[:shift_index])
    values[:shift_index] = np.minimum(uniform, barankin_bound)
    
    cr_bound = np.pi / (omega_bw * duration * omega0**2 * snr[shift_index:])
    uniform_sr = 1./(12.0 * sample_rate**2) #* np.ones(snr[shift_index:].shape)
    values[shift_index:] = np.maximum(uniform_sr, np.minimum(uniform, cr_bound))
    
    return values

def richards_function(snr : np.ndarray, delay, sample_rate, snr0, Q, B, nu):
    """Compute richard's curve to approximate the variance of error made.

    Args:
        snr (_type_): _description_
        delay (_type_): _description_
        sample_rate (_type_): _description_
        snr0 (_type_): _description_
        Q (_type_): _description_
        B (_type_): _description_
        nu (_type_): _description_

    Returns:
        _type_: _description_
    """
    A = delay**2/12.0
    K = 1./(12.0 * sample_rate**2)
    denominator = (1 + Q * np.exp(-B*(snr-snr0)))**(1./nu)
    return A + (K-A)/denominator

def match_richards(target_values, all_snr, delay, sample_rate):
    def cost_function(params):
        snr0, Q, B, nu = params
        richard_values = richards_function(all_snr, delay, sample_rate, snr0, Q, B, nu)
        # MSE cost
        cost = mse_cost(richard_values, target_values)
        return cost
    initial_params = [1., 1., 1., 1.]
    results = minimize(cost_function, initial_params, tol = 1e-28)
    
    snr0, Q, B, nu = results.x
    return snr0, Q, B, nu

def match_ideal(target_values, all_snr, delay, omega0, omega_bw, sample_rate):
    def cost_function(snr_shift):
        ideal_values = ideal_function(all_snr, snr_shift, omega0, omega_bw, duration, delay, sample_rate)
        # RMSE cost
        cost = mse_cost(ideal_values, target_values)
        return cost
    #snr_shift0 = 1.0
    #results = minimize(cost_function, snr_shift0, tol = 1e-28, method='Nelder-Mead', options={'adaptive': True})
    #print(f'result ideal : {results}')
    #snr_shift = results.x
    all_costs = []
    for snr_threshold in all_snr:
        all_costs.append(cost_function(snr_threshold))
    snr_shift = all_snr[np.argmin(all_costs)]
            
    return snr_shift
    

def ziv_zakai_plot(
    bandwidth : float,
    duration : float,
    central_frequency : float,
    environment : Environment,
    sample_rate = 384000,
    snr_min : float = 0,
    snr_max : float = 5,
    nb_points : int = 100,
    max_nb_draws : int = 100,
    save_path : Optional[str] = None,
    noises_folder : Optional[str] = None,
    position : np.ndarray = np.zeros(3),
    save_path_np : Optional[str] = None,
    nb_samples_fs : int = 5,
    load_path : str = 'None'
    ):
    
    delay_seconds = 2 * next(iter(environment.tetrahedras.values())).max_delay_seconds
    
    omega_bw = 2*np.pi * bandwidth
    omega0 = central_frequency * 2*np.pi
    
    snr1 = snr1_value(omega_bw, duration, omega0, delay_seconds)
    snr2 = snr2_value(omega_bw, duration, omega0)
    snr3 = snr3_value(omega_bw, duration, omega0)
    
    if noises_folder is not None:
        list_noises = load_noises(noises_folder, 384000)
        filter_params = PreFilterParameters(4, 'butter_bp')
        # ICI : FILTRER LES NOISES PAR BUTTERWORTH
        for i in range(len(list_noises)):
            list_noises[i] = filter_data_array(list_noises[i].T, filter_params, sample_rate, 'Whistle').T
        nb_draws = min(max_nb_draws,len(list_noises))
        pure_beluga_noise = generate_gaussian_whistle(
            central_frequency,
            duration,
            f0 = central_frequency - bandwidth/2,
            f1 = central_frequency + bandwidth/2,
            sample_rate=sample_rate,
            output_path= None,
            show_fig=False
        )
        pure_beluga_noise += generate_gaussian_whistle(
            2*central_frequency,
            duration,
            f0 = 2*central_frequency - bandwidth/2,
            f1 = 2*central_frequency + bandwidth/2,
            sample_rate=sample_rate,
            output_path= None,
            show_fig=False
        )
        pure_beluga_noise += generate_gaussian_whistle(
            3*central_frequency,
            duration,
            f0 = 3*central_frequency - bandwidth/2,
            f1 = 3*central_frequency + bandwidth/2,
            sample_rate=sample_rate,
            output_path= None,
            show_fig=False
        )
        pure_beluga_noise = filter_data_array(pure_beluga_noise.T, filter_params, sample_rate, 'Whistle').T
    else :
        list_noises = None
        nb_draws = None
        pure_beluga_noise = None
    
    snr_values = np.logspace(snr_min, snr_max, nb_points)
    error_values = np.zeros(snr_values.shape)
    monte_carlo_values = np.zeros(snr_values.shape)
    time0 = time()
    for i in range(len(error_values)):
        error_values[i] = ziv_zakai_value(snr_values[i], bandwidth, duration, central_frequency,environment,sample_rate)
        if not os.path.exists(load_path):
            if noises_folder is not None and nb_draws is not None and list_noises is not None and pure_beluga_noise is not None: 
                monte_carlo_values[i] = monte_carlo_error(snr_values[i], environment, nb_draws, list_noises, pure_beluga_noise, position = position, nb_samples_fs = nb_samples_fs)
            print(f'Computed {(i+1)}/{len(error_values)} values ({float((i+1))/len(error_values) * 100 :.2f}%) in {time()-time0:.1f}s ({(len(error_values)-float((i+1)))/float((i+1)) * (time()-time0):.1f}s remaining)')
    
    if os.path.exists(load_path):
        monte_carlo_values = np.load(load_path)
    
    if save_path_np is not None:
        np.save(save_path_np, np.array(monte_carlo_values))

    # ICI ON A TOUT : FAIRE MATCHER LES COURBES
    
    snr_shift = match_ideal(monte_carlo_values, snr_values, delay_seconds, omega0, omega_bw, sample_rate)
    ideal_curve = np.zeros(snr_values.shape)
    print(f'SNR Shift : {snr_shift}')
    
    snr0 , Q, B, nu = match_richards(monte_carlo_values, snr_values, delay_seconds, sample_rate)
    richard_curve = np.zeros(snr_values.shape)
    print(f'SNR 0 : {snr0}, Q : {Q}, B : {B}, nu : {nu}')
    
    for i in range(len(snr_values)):
        richard_curve[i] = richards_function(snr_values[i], delay_seconds, sample_rate, snr0, Q, B, nu)
        
    ideal_curve = ideal_function(snr_values, snr_shift, omega0, omega_bw, duration, delay_seconds, sample_rate)
    
    crb_values = np.pi / (omega_bw * duration * omega0**2 * snr_values)
    barankin_values = 12.0 * np.pi /(omega_bw**3 * duration * snr_values)
    transition_bound_values = np.zeros(snr_values.shape)
    for i in range(len(transition_bound_values)):
        transition_bound_values[i] = transition_bound(omega_bw, omega0, snr_values[i], delay_seconds, duration)
    
    mse_zzb = mse_cost(error_values, monte_carlo_values)
    mse_ideal = mse_cost(ideal_curve, monte_carlo_values)
    mse_richard = mse_cost(richard_curve, monte_carlo_values)
    
    plt.figure(figsize=(8, 6))
    plt.plot(10*np.log10(snr_values), error_values, label='Ziv Zakai bound', color='black', linestyle = "-", linewidth=1.0 )
    plt.plot(10*np.log10(snr_values), ideal_curve, label='Ideal curve', color='tab:blue', linestyle = "-", linewidth=1.0 )
    plt.plot(10*np.log10(snr_values), richard_curve, label='Richard curve', color='tab:orange', linestyle = "-", linewidth=1.0 )
    

    plt.plot(10*np.log10(snr_values), crb_values, label='Cramèr Rao bound', color='tab:orange', linestyle = "--", linewidth = 0.7, alpha = 0.7)
    plt.plot(10*np.log10(snr_values), transition_bound_values, label='Transition', color='tab:blue', linestyle = "--", linewidth = 0.7,alpha = 0.7)
    plt.plot(10*np.log10(snr_values), barankin_values, label='Barankin bound', color='tab:green', linestyle = "--", linewidth = 0.7,alpha = 0.7)
    plt.plot(10*np.log10(snr_values), monte_carlo_values , label='Monte Carlo simulation', color='tab:red', linestyle = "-", linewidth = 1.0)
    
    plt.axhline(y = delay_seconds**2/12.0, color='tab:grey', linestyle=':', linewidth=0.7, label = "Uniform random bound")
    plt.axhline(y = 1./sample_rate**2 / 12.0, color='tab:grey', linestyle='-.', linewidth=0.7, label = "Sample rate bound")
    
    plt.axvline(x=10*np.log10(snr1), color='tab:grey', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.axvline(x=10*np.log10(snr2), color='tab:grey', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.axvline(x=10*np.log10(snr3), color='tab:grey', linestyle='--', linewidth=0.7, alpha=0.7)
    
    plt.text(10*np.log10(snr1), 1E-14, 'SNR1', horizontalalignment='center')
    plt.text(10*np.log10(snr2), 1E-14, 'SNR2', horizontalalignment='center')
    plt.text(10*np.log10(snr3), 1E-14, 'SNR3', horizontalalignment='center')
        
    plt.xlabel('SNR (dB)')
    plt.ylabel('Variance of error (log scale)')
    plt.yscale('log')
    #plt.xscale('log')
    plt.legend()
    plt.title(f'Theoretical error bound\nfc = {central_frequency}Hz\nbw = {bandwidth}Hz')#\nnb points : {nb_points} - nb tdoa draws : {nb_samples_fs}')
    plt.grid(False)
    
    if save_path is not None:
        plt.savefig(save_path)
    #plt.show()
    plt.close()
    return [snr_shift, snr1, snr2, snr3], [snr0 , Q, B, nu], [mse_zzb, mse_richard, mse_ideal]
    
def plot_snr_shifts(frequencies, snrs):
    plt.figure(figsize=(12, 8))
    all_snrs = np.array(snrs).T
    snr_names = ['SNR Shift', 'SNR1', 'SNR2', 'SNR3']
    for snr_list, snr_name in zip(all_snrs, snr_names):
        plt.plot(frequencies, 10*np.log10(snr_list), marker='o', label=snr_name)
    plt.plot(frequencies, 5*np.log10(all_snrs[2]) +  5*np.log10(all_snrs[3]), marker='o', label='Average SNR2 - SNR3')
    plt.title('SNR Shifts vs Central Frequencies')
    plt.xlabel('Central frequency [Hz]')
    plt.ylabel('SNR Shift [dB]')
    plt.legend()
    plt.grid(False)
    plt.savefig("test_data/zivzakai_outputs/results_snr_shifts.png")
    plt.show()

def plot_mse_vs_frequencies(frequencies, all_mses):
    plt.figure(figsize=(12, 4))
    all_mses = np.sqrt(np.array(all_mses).T)
    mse_names = ['Ziv Zakai bound', 'Richard', 'Ideal']
    for mse_list, mse_name in zip(all_mses, mse_names):
        plt.plot(frequencies, np.log10(mse_list), marker='o', label=mse_name)
    plt.title('MSEs vs Frequencies')
    plt.xlabel('Central frequency [Hz]')
    plt.ylabel('log10 RMSE')
    plt.legend()
    plt.grid(False)
    plt.savefig("test_data/zivzakai_outputs/results_mse.png")
    plt.show()

def plot_richard_params(frequencies, richard_params):
    plt.figure(figsize=(12, 4))
    richard_params = np.array(richard_params).T
    richard_names = ['snr 0', 'Q', 'B', 'nu']
    for richard_param_list, richard_name in zip(richard_params, richard_names):
        plt.plot(frequencies, richard_param_list, marker='o', label=richard_name)
    plt.title('Richard Parameters vs Frequencies')
    plt.xlabel('Central frequency [Hz]')
    plt.ylabel('Richard Parameters')
    plt.legend()
    plt.grid(False)
    plt.savefig("test_data/zivzakai_outputs/results_richard.png")
    plt.show()

def zz_results_plot(output_list):
    frequencies, snrs, all_mses, richard_params = output_list
    plot_snr_shifts(frequencies, snrs)
    plot_mse_vs_frequencies(frequencies, all_mses)
    plot_richard_params(frequencies, richard_params)


def extract_frequency(filename):
    # Utiliser une expression régulière pour extraire la valeur
    pattern = r'tdoadraws_(\d+\.?\d*)fc'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None
    
if __name__== '__main__':
    environment = Environment('jsons/environments/env_cacouna.json',True)
    bandwidth = 300
    duration = 1.0
    npy_folder = os.listdir("test_data/zivzakai_outputs")
    central_frequencies = np.unique([float(extract_frequency(file)) for file in npy_folder if (file.endswith('.npy') and extract_frequency(file))])
    nb_points = 20
    central_frequencies = [750.0, 1000.0, 3000.0, 10000.0, 18000.0]
    pos = np.zeros(3)
    #nb_samples_fs = 8 # Discretises the -fs/2, fs/2 segment to reach approx fs^2/12 in average
    nb_samples_fs = 1
    noises_folder = 'noises'
    central_freq_used = []
    snr_shifts = []
    richard_params = []
    all_mses = []
    if not os.path.isfile("test_data/zivzakai_outputs/outputs_pkl.pkl"):
        for central_frequency in central_frequencies:
            print(f'central_frequency : {central_frequency}')
            save_path_zzp = f"test_data/zivzakai_outputs/zivzakaiplot_3pos_{nb_points}points_{nb_samples_fs}tdoadraws_{central_frequency}fc_{int(bandwidth)}bw.png"
            save_path_numpy =f"test_data/zivzakai_outputs/zivzakaiplot_3pos_{nb_points}points_{nb_samples_fs}tdoadraws_{central_frequency}fc_{int(bandwidth)}bw.npy"
            load_path = save_path_numpy
            snrs, richard_param, mses = ziv_zakai_plot(bandwidth, duration, central_frequency,environment, nb_points = nb_points, save_path = save_path_zzp, noises_folder= noises_folder, position = pos, save_path_np = save_path_numpy, nb_samples_fs = nb_samples_fs, load_path = load_path)
            snr_shifts.append(snrs)
            richard_params.append(richard_param)
            all_mses.append(mses)
            central_freq_used.append(central_frequency)
        output_list = [central_freq_used, snr_shifts, all_mses, richard_params]
        with open("test_data/zivzakai_outputs/outputs_pkl.pkl", 'wb') as fichier_pickle:
            pickle.dump(output_list, fichier_pickle)
    else:
        with open("test_data/zivzakai_outputs/outputs_pkl.pkl", 'rb') as fichier_pickle:
            output_list = pickle.load(fichier_pickle)
    zz_results_plot(output_list)