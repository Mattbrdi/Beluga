"""Code to generate audio data from an environment, a whistle,
eventually a boat or other parasitic auto-corrrelated noise, and a position groundtruth."""

from src.utils.sub_classes import AudioArray, Environment, Tetrahedra, AudioMetadata
from src.utils.wav_parser import read_mono_file, read_wav_file, output_wav_file
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import scipy.signal as sig


######################################
##### Noise generating functions #####
######################################

# We could also just add a window with no beluga sound

class FlowNoise:
    """Class to generate Gaussian hydrophone noise."""
    def __init__(self, f_c=100, order=2):
        """
        Initialize a FlowNoise object.

        Parameters:
        - f_c: Cutoff frequency.
        - order: Order of the filter.
        """
        self.f_c = f_c  # Cutoff frequency
        self.order = order  # Order of the filter

    def generate_noise(self, sample_rate, samples, nb_canals):
        """
        Generate flow noise.

        Parameters:
        - sample_rate: Sample rate of the audio.
        - samples: Number of samples.
        - noise_shape: Shape of the noise.

        Returns:
        - flow_noise: Generated flow noise.
        """
        white_noise = np.random.normal(0, 1, (nb_canals, samples))
        b, a = sig.butter(self.order, self.f_c / (sample_rate / 2), btype='low', analog=False)
        flow_noise = sig.filtfilt(b, a, white_noise, axis=1)
        return flow_noise

class MechanicalNoise:
    """Class to generate mechanical noise."""
    def __init__(self):
        """
        Initialize a MechanicalNoise object.
        """
        m = 0.01  # Mass of the sensitive element (kg)
        k = 1000  # Stiffness (N/m)
        f0 = np.sqrt(k / m) / (2 * np.pi)  # Resonance frequency
        self.lowcut = f0 * 0.8  # Low cutoff frequency
        self.highcut = f0 * 1.2  # High cutoff frequency
        self.order = 2  # Order of the filter

    def generate_noise(self, sample_rate, samples, nb_canals):
        """
        Generate mechanical noise.

        Parameters:
        - sample_rate: Sample rate of the audio.
        - samples: Number of samples.
        - noise_shape: Shape of the noise.

        Returns:
        - mecha_noise: Generated mechanical noise.
        """
        b, a = sig.butter(self.order, [self.lowcut / (sample_rate / 2), self.highcut / (sample_rate / 2)], btype='band')
        mecha_noise = sig.filtfilt(b, a, np.random.normal(0, 1, (nb_canals, samples)))
        return mecha_noise

class ThermalNoise:
    """Class to generate thermal noise."""
    def __init__(self):
        """
        Initialize a ThermalNoise object.
        """
        self.kB = 1.38e-23  # Boltzmann constant (J/K)
        self.T = 300  # Temperature in Kelvin
        self.R = 1000  # Hydrophone resistance (Ohms)

    def generate_noise(self, sample_rate, samples, noise_shape=1):
        """
        Generate thermal noise.

        Parameters:
        - sample_rate: Sample rate of the audio.
        - samples: Number of samples.
        - noise_shape: Shape of the noise.

        Returns:
        - thermal_noise: Generated thermal noise.
        """
        B = sample_rate / 2  # Assumed bandwidth (Nyquist)
        sigma_V = np.sqrt(4 * self.kB * self.T * self.R * B)
        return np.random.normal(0, sigma_V, (noise_shape, samples))

class SelfNoise:
    """Class to generate self-noise."""
    def __init__(self):
        """
        Initialize a SelfNoise object.
        """
        self.flow_noise = FlowNoise()
        self.mechanical_noise = MechanicalNoise()
        self.thermal_noise = ThermalNoise()

    def generate_noise(self, sample_rate, samples: int, nb_canals : int):
        """
        Generate self-noise.

        Parameters:
        - sample_rate: Sample rate of the audio.
        - samples: Number of samples.
        - noise_shape: Shape of the noise.

        Returns:
        - signal: Generated self-noise.
        """
        signal = np.zeros((nb_canals, samples))
        signal += self.flow_noise.generate_noise(sample_rate, samples, nb_canals)
        signal += self.mechanical_noise.generate_noise(sample_rate, samples, nb_canals)
        signal += self.thermal_noise.generate_noise(sample_rate, samples, nb_canals)
        return signal

###########################################
##### Auxiliary computation functions #####
###########################################

def generate_gaussian_whistle(center_frequency = 1E4,
                              audio_length = 1.0,
                              whistle_begin = 0.15,
                              whistle_end = 0.85,
                              enveloppe_std = 0.7/6,
                              f0 : float = 8500,
                              f1 :float = 11500,
                              sample_rate = 384000,
                              output_path = None,
                              show_fig = False):
    
    """Generate a four canal synchronised whistle modelled as
    x(t) = exp(-(t-T/2)^2/(2 sigma^2)) sin(2 pi fc t)

    Args:
        center_frequency (_type_, optional): fc. Defaults to 1E4.
        audio_length (float, optional): T. Defaults to 1.0.
        whistle_begin (float, optional): t0. Defaults to 0.15.
        whistle_end (float, optional): tf. Defaults to 0.85.
        enveloppe_std (_type_, optional):sigma. Defaults to (tf - t0)/6.
        sample_rate (int, optional): Sample rate. Defaults to 384000.
        output_path (_type_, optional): Path to save wav. Defaults to None if no output desired.
        show_fig (bool, optional): True if spectrogram plot desired. Defaults to False.
    """
    # General parameters
    t = np.linspace(0, audio_length, int(sample_rate * audio_length), endpoint=False)
    
    # Whistle generation
    whistle_duration = whistle_end - whistle_begin
    t_whistle = np.linspace(-whistle_duration/2, whistle_duration/2, int(sample_rate * whistle_duration), endpoint=False)
    """ # Gaussian enveloppe
    x_whistle = np.sin(2 * np.pi * center_frequency * t_whistle)
    tc = whistle_duration / 2
    envelope = np.exp(-0.5 * ((t_whistle - tc) / enveloppe_std) ** 2)
    x_whistle *= envelope
    """
    x_whistle = (f1-f0) * np.sinc((f1-f0)*t_whistle) * np.cos(np.pi * t_whistle*(f0 + f1))
    x_whistle/=np.max(np.abs(x_whistle))
    # Mapping it to the full signal
    x_total = np.zeros_like(t)
    start = int(whistle_begin * sample_rate)
    x_total[start:start + len(x_whistle)] = x_whistle
    #x_total += 0.001*(np.random.rand(len(x_total))-0.5)
    tetra_array = np.array([x_total for _ in range(4)])
    
    if output_path is not None:
        output_wav_file(tetra_array.T, sample_rate, output_path)
    
    if show_fig:
        plt.figure(figsize=(10, 4))
        plt.specgram(x_total, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='viridis')
        plt.title("Spectrogramme du sifflement pur à 10 kHz")
        plt.xlabel("Temps (s)")
        plt.ylabel("Fréquence (Hz)")
        plt.colorbar(label="Amplitude (dB)")
        plt.tight_layout()
        plt.show()
        
    return tetra_array
        

def travel_times(tetrahedra : Tetrahedra, water_sound_speed : float, beluga_enu : np.ndarray):
    """
    Calculate travel time from beluga to hydrophones.

    Parameters:
    - beluga_enu: Position of the beluga in enu coordinates.

    Returns:
    - travel_time: Travel time array.
    """
    hydrophones_enu_pos = tetrahedra.rotated_hydro_pos_enu
    
    return 1. / water_sound_speed * np.linalg.norm(beluga_enu - hydrophones_enu_pos, axis=1)

def tdoa_from_travel_time(travel_times : np.ndarray, tetra_id : str)->dict[str, float]:
    tdoas = {}
    for i in range(4):
        for j in range(i+1,4):
            tdoas[f'{tetra_id}_H{j+1}-H{i+1}'] = travel_times[j] - travel_times[i]
    return tdoas

def generate_four_can(autocorr_audio : np.ndarray, tdoas_from_h1 : np.ndarray, real_start_time : float, frame_rate:int, audio_length:float):
    """Generate a four canal array with an isolated noise.

    Args:
        autocorr_audio (np.ndarray): 4 canal synchronized audio to delay at each tdoas
        tdoas_from_h1 (np.ndarray): 4 tdoas corresponding to each pair of hydrophones, including H1H1 first
        real_start_time (float): Padding de départ - min(tdoas)
        frame_rate (int): Echantillonnage
        audio_length (float): Duree de l'audio à générer en secondes

    Returns:
        data_array (np.ndarray): Audio 4 canaux durant audio_len 
    """
    idx_length = int(frame_rate*audio_length)
    data_array = np.zeros((4, idx_length))
    for i, tdoa in enumerate(tdoas_from_h1):
        begin_idx = int((real_start_time + tdoa ) * frame_rate)
        max_idx = min(idx_length, np.shape(autocorr_audio)[1])
        data_array[i, begin_idx:max_idx] = autocorr_audio[i,:max_idx-begin_idx]
    return data_array

def generate_correlated_array(
    environment : Environment,
    start_time : float,
    pos_enu : np.ndarray,
    audio_wav_path : str,
    audio_length : float,
    ):
    ##### Computing travel times and TDOAS to each array #####
    all_travel_times = [travel_times(tetra, environment.sound_speed, pos_enu) for tetra in environment.tetrahedras.values()]
    all_tdoas = [tdoa_from_travel_time(travel_time,tetra_id) for travel_time, tetra_id in zip(all_travel_times, environment.tetrahedras.keys())]
    real_start_times = [start_time - min(0,np.min(list(tdoa_array.values()))) for tdoa_array in all_tdoas] # The time such that real start time + tdoa > 0
    tdoas_from_h1 = np.array([[0]+list(tdoas.values())[0:3] for tdoas in all_tdoas])
    
    ##### Parsing the auto-correlated audio file #####
    audio_data, frame_rate = read_wav_file(audio_wav_path)

    ##### Creating arrays with the whistle #####
    output_data_arrays = [generate_four_can(audio_data, tdoa, real_start_time, frame_rate, audio_length) for tdoa, real_start_time in zip(tdoas_from_h1,real_start_times)]
    
    return output_data_arrays, all_tdoas, frame_rate

def generate_correlated_array_from_nparray(
    environment : Environment,
    start_time : float,
    pos_enu : np.ndarray,
    audio_data : np.ndarray,
    frame_rate : int,
    audio_length : float,
    ):
    ##### Computing travel times and TDOAS to each array #####
    all_travel_times = [travel_times(tetra, environment.sound_speed, pos_enu) for tetra in environment.tetrahedras.values()]
    all_tdoas = [tdoa_from_travel_time(travel_time,tetra_id) for travel_time, tetra_id in zip(all_travel_times, environment.tetrahedras.keys())]
    real_start_times = [start_time - min(0,np.min(list(tdoa_array.values()))) for tdoa_array in all_tdoas] # The time such that real start time + tdoa > 0
    tdoas_from_h1 = np.array([[0]+list(tdoas.values())[0:3] for tdoas in all_tdoas])

    ##### Creating arrays with the whistle #####
    output_data_arrays = [generate_four_can(audio_data, tdoa, real_start_time, frame_rate, audio_length) for tdoa, real_start_time in zip(tdoas_from_h1,real_start_times)]
    
    return output_data_arrays, all_tdoas, frame_rate

def generate_correlated_array_from_nparray_int_tdoas(
    environment : Environment,
    start_time : float,
    pos_enu : np.ndarray,
    audio_data : np.ndarray,
    frame_rate : int,
    audio_length : float,
    ):
    ##### Computing travel times and TDOAS to each array #####
    all_travel_times = [travel_times(tetra, environment.sound_speed, pos_enu) for tetra in environment.tetrahedras.values()]
    all_tdoas = [tdoa_from_travel_time(travel_time,tetra_id) for travel_time, tetra_id in zip(all_travel_times, environment.tetrahedras.keys())]
    """
    for i , tdoa in enumerate(all_tdoas):
        for key , value in tdoa.items():
            all_tdoas[i][key] = int(value*frame_rate)/float(frame_rate) + np.random.uniform(0., 1.0/(2.0*frame_rate)) #np.random.uniform(-1.0/(2.0*frame_rate), 1.0/(2.0*frame_rate)) Putting 0 should be enough
    """
    real_start_times = [start_time - min(0,np.min(list(tdoa_array.values()))) for tdoa_array in all_tdoas] # The time such that real start time + tdoa > 0
    tdoas_from_h1 = np.array([[0]+list(tdoas.values())[0:3] for tdoas in all_tdoas])

    ##### Creating arrays with the whistle #####
    output_data_arrays = [generate_four_can(audio_data, tdoa, real_start_time, frame_rate, audio_length) for tdoa, real_start_time in zip(tdoas_from_h1,real_start_times)]
    
    return output_data_arrays, all_tdoas, frame_rate
    
    
##########################
##### Main function ######
########################## 

def all_arrays_from_pos(
    whistle_wav_path : str,
    whistle_pos_enu : np.ndarray,
    environment : Environment,
    snr_power : list[list[float]],
    audio_length : float = 1.0,
    start_time : float = 0.0,
    corr_noise_wav_path : Optional[str] = None,
    corr_noise_pos_enu : Optional[np.ndarray] = None,
    non_corr_noise_path : Optional[str] = None,
    ) -> tuple[list[AudioArray], dict] :
    
    """Function to generate audio_arrays with known snr, pos, tdoas from a 'clean' whistle.

    Args:
        whistle_wav_path (str): Path to the 'isolated' whistle
        whistle_pos_enu (np.ndarray) : Pos of this whistle
        environment (Environment): Environnement loadé
        snr_power (list[list[float]]): Liste de SNR Powers à chaque tétrahèdre (POWERS ET PAS DB) 
        audio_length (float, optional): Durée de l'audio à générer. Defaults to 1.0.
        start_time (float, optional): Offset le début du whistle sur les audios (sinon le premier commence à t = 0). Defaults to 0.0.
        noise_wav_path (Optional[str], optional): Path pour load un bruit auto corrélé. Defaults to None.
        noise_pos_enu (np.ndarray) : Position de ce bruit auto corrélé

    Returns:
        audio_arrays(list[AudioArray]): Liste des AudioArray générés à chaque tétrahèdre
        gtruth_dict(dict): Dictionnaire contenant les vrais tdoas et les vraies positions
    """
    nb_tetrahedras = len(environment.tetrahedras)
    ##### Creating the data arrays with beluga whistles #####
    whistle_data_arrays, whistle_tdoas, frame_rate, = generate_correlated_array(
        environment,
        start_time,
        whistle_pos_enu,
        whistle_wav_path,
        audio_length
    )
    
    ##### Adding uncorrelated noise #####
    if non_corr_noise_path is None:
        self_noise_module = SelfNoise()
        non_corr_noise_array = self_noise_module.generate_noise(frame_rate, int(audio_length*frame_rate), 4)
    else:
        non_corr_noise_array , non_corr_rate = read_wav_file(non_corr_noise_path)
        
        ##### Quick troubleshooting #####
        if non_corr_rate != frame_rate:
            raise ValueError('Non correlated noise frame rate is different from whistle frame rate.')
        if non_corr_noise_array.shape[1] < int(audio_length*frame_rate):
            raise ValueError(f'The non correlated noise is too short.\nExpected at least {audio_length:.2f}s but got {float(non_corr_noise_array.shape[1])/frame_rate:.2f}')
        #################################
        
        non_corr_noise_array = non_corr_noise_array[:,:int(audio_length*frame_rate)]
        
    noise_arrays = nb_tetrahedras * [non_corr_noise_array] # Reference copy, it's the same
    
    ##### Creating the data arrays with correlated noise #####
    corr_noise_tdoas = None
    if corr_noise_wav_path is not None and corr_noise_pos_enu is not None:
        corr_noise_data, corr_noise_tdoas, corr_noise_frame_rate = generate_correlated_array(
        environment,
        start_time,
        corr_noise_pos_enu,
        corr_noise_wav_path,
        audio_length
    )
        if corr_noise_frame_rate != frame_rate :
            raise ValueError(f'The noisy frame rate is {corr_noise_frame_rate}Hz,\ndifferent from the whistle frame rate of {frame_rate}Hz')
        
        noise_arrays = [noise_array + corr_noise for noise_array, corr_noise in zip(noise_arrays, corr_noise_data)]
    
    ##### Adjusting SNR #####
    noise_array_powers = [np.sum(noise_array**2, axis = 1) for noise_array in noise_arrays]
    whistle_array_powers = [np.sum(whistle_data_array**2, axis = 1) for whistle_data_array in whistle_data_arrays]
    old_maxes = [np.max(np.abs(whistle), axis=1, keepdims = True) for whistle in whistle_data_arrays]
    
    for i, (snr_list, whistle_power_list, noise_power_list) in enumerate(zip(snr_power, whistle_array_powers, noise_array_powers)):
        correctors = np.array(snr_list) * np.array(noise_power_list)/np.array(whistle_power_list)
        correctors = correctors.reshape((4,1))
        whistle_data_arrays[i] *= np.sqrt(correctors)
    
    final_arrays = [whistle_data_array + noise_data_array for whistle_data_array, noise_data_array in zip(whistle_data_arrays, noise_arrays)]
    
    ##### Make sure to stay between [-1,1] #####
    for i in range(len(final_arrays)):
        final_arrays[i] = final_arrays[i] / np.max(np.abs(final_arrays[i])) * old_maxes[i]
    
    ##### Parsing the to AudioArray class #####
    audio_metadatas = [AudioMetadata(tetra_id,'Whistle',audio_length,start_time,snr_pow, frame_rate, (500,20000),10000) for tetra_id, snr_pow in zip(environment.tetrahedras.keys(),snr_power)]
    audio_arrays = [AudioArray(audio_metadata, tetra_data, True, final_array) for audio_metadata, tetra_data, final_array in zip(audio_metadatas,environment.tetrahedras.values(), final_arrays)]
    
    ##### Generating the groundtruth output #####
    if len(snr_power) != len(whistle_tdoas):
        print(f'WARNING : not the same amount of SNR and TDOAs')
    if nb_tetrahedras != len(snr_power):
        print(f"WARNING : Number of SNRs and number of tetrahedras don't match")

    
    groundtruth_dict = {
        'Whistle tdoas' : whistle_tdoas,
        'Correlated noise tdoas' : corr_noise_tdoas,
        'Beluga pos ENU' : list(whistle_pos_enu),
        'Corr noise pos ENU' : corr_noise_pos_enu,
        'SNR power' : snr_power,
        'ENU ref' : environment.enu_ref
    }
    
    return audio_arrays , groundtruth_dict
