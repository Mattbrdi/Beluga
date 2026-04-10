"""Code for TDOA and CRB estimation."""

from src.utils.sub_classes import AudioArray, HydrophonePair, AudioMetadata
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import correlate, find_peaks, peak_widths
from scipy.special import erfc, ndtri
from scipy.optimize import root_scalar
from typing import Optional

############################
##### TDOA computation #####
############################

def compute_correlation_score(correlation_val : np.ndarray, sample_rate):
    """Compute the correlation score for one hydrophone pair."""

    correlation_values = correlation_val / np.max(np.abs(correlation_val))  # Normalization
    
    # Pics
    prominence = 0.05 * (np.max(correlation_values) - np.min(correlation_values))
    peaks, props = find_peaks(correlation_values, prominence=prominence)
    
    num_peaks = len(peaks)
    score_text = f'Scores :\nPeaks : {num_peaks}'
    
    # Prominence to noise ratio
    std_corr = np.std(correlation_values)
    score_text += f'\nstd : {std_corr:.2f}'
    
    if len(peaks) >= 1:
        # Hauteurs des pics
        heights = correlation_values[peaks]

        # Tri décroissant des hauteurs
        sorted_indices = np.argsort(heights)[::-1]
        top_index = sorted_indices[0]
        max_height = heights[top_index]
        # Largeur à mi-hauteur du pic le plus haut
        results_half = peak_widths(correlation_values, peaks, rel_height=0.5)
        widths = results_half[0]
        width_max_peak = widths[top_index]
        score_text += f'\nMax peak width : {width_max_peak/sample_rate*10E6:.0f}µs'
        sharpness = props['prominences'][top_index] / width_max_peak
        score_text += f'\nPeak sharpness : {sharpness:.2f}'

        
        # Différence entre le 1er et le 2e plus haut pic
        if len(peaks) >= 2:
            second_max_height = heights[sorted_indices[1]]
            height_diff = max_height - second_max_height
            score_text += f'\nHeight diff : {height_diff:.2f}'
            if second_max_height >0:
                height_ratio = max_height / second_max_height
                score_text += f'\nHeight ratio : {height_ratio:.2f}'
            else:
                height_ratio = np.inf
                score_text += f'\nHeight ratio : inf'
        else:
            height_diff = None
            height_ratio = None
    else:
        sharpness = None
        height_ratio = None
        width_max_peak = None
        height_diff = None
        widths = None
        
    scores = [std_corr, num_peaks, width_max_peak, height_diff, height_ratio, sharpness]
    return scores, score_text

def gcc_phat_from_pair(audio_one, audio_two):

    if len(audio_one) != len(audio_two):
        raise ValueError("Audio signals must have the same length for GCC-PHAT.")
    
    fft_one = fft(audio_one, 2*len(audio_one) - 1)
    fft_two = fft(audio_two, 2*len(audio_one) - 1)
    
    cross_corr_spectrum = fft_one * np.conj(fft_two)
    phat_function = 1./np.abs(cross_corr_spectrum)
    gcc_phat = ifft(cross_corr_spectrum * phat_function).real
    
    return np.roll(gcc_phat, len(audio_one) - 1)  # Shift to center the zero lag
    

def tdoa_from_pair(hydrophone_pair: HydrophonePair, sample_rate, use_gcc: bool, compute_scores : bool):
    """
    Calculate Time Difference of Arrival (TDOA) from a pair of hydrophones.

    Parameters:
    - hydrophone_pair: Pair of hydrophones.
    - sample_rate: Sampling rate of the audio signals.

    Returns:
    - tdoa: Time Difference of Arrival.
    """
    use_tdoa = True
    audio_one = hydrophone_pair.hydrophone_ref.audio_r
    audio_two = hydrophone_pair.hydrophone_delta.audio_r
    max_delay = hydrophone_pair.max_delay_idx
    
    ############################################################
    if not use_gcc:
        cross_corr = correlate(audio_one, audio_two, mode='full')
    else:
        cross_corr = gcc_phat_from_pair(audio_one, audio_two)
    ############################################################
    
    normalized_cross_corr = cross_corr/(np.max(np.abs(cross_corr))+1E-34)  # Avoid division by zero
    lowest_index = int((2 * len(hydrophone_pair.hydrophone_ref.audio_r) - 1)/2 - max_delay)
    upper_index = int((2 * len(hydrophone_pair.hydrophone_ref.audio_r) - 1)/2 + max_delay)
    processed_correlation = normalized_cross_corr[lowest_index:upper_index]
    
    if compute_scores:
        scores, _ = compute_correlation_score(processed_correlation, sample_rate)
    else:
        scores = None
    
    argmax_idx = np.argmax(processed_correlation)
    tdoa = (argmax_idx - (len(audio_one)-1) + lowest_index)/sample_rate

    if argmax_idx <2 or argmax_idx > (len(processed_correlation)-2) :#or processed_correlation[argmax_idx]<0:
        #print("WARNING : Some TDOAs are at the frontier or the correlation is <0")
        # TODO put back the warning after benchmarking
        use_tdoa = False
    #if np.abs(tdoa) < 1E-9:
        # Avoid having 0
    #    tdoa = 1E-9
    return tdoa, use_tdoa, scores

############################
##### Error estimation #####
############################

def phi(y):
    return 0.5 * erfc(y/np.sqrt(2))

def phi_inv(y):
    return ndtri(1-y)


def transition_bound(omega_bw, omega0, snr, delay, duration):
    """Calculate the value of minimal error in the transition phase."""

    beta_squared = (omega_bw * duration * omega0**2 * snr) / (4 * np.pi)
    eta_squared = (omega_bw**3 * duration * snr) / (48 * np.pi)
    eta = np.sqrt(eta_squared)
    f_0 = omega0 / (2.0 * np.pi)
    
    e_bar_squared = 1.0/(2*eta_squared) * phi(eta/f_0) + 1.0/(2*beta_squared) * (0.5 - phi(eta/f_0))
    
    return e_bar_squared

def snr1_value(omega_bw, duration, omega0, delay):
    return 72.0/(omega_bw * duration / (2*np.pi)) * (omega0/omega_bw)**2 * 1.0/((omega0 * delay)**2)

def snr2_value(omega_bw, duration, omega0):
    return 2.17/(np.pi*(omega_bw * duration / 2)) * (omega0/omega_bw)**2

def snr3_value(omega_bw, duration, omega0):
    return 6.0/(np.pi*(omega_bw * duration / 2)) * (omega0/omega_bw)**2 * (phi_inv(omega_bw**2/(24*omega0**2)))**2

def capped_crb(delay, omega_bw, duration, omega0, snr, sample_rate, threshold = 1E-10):
    uniform_delay = delay**2/12.
    cr_bound = np.pi / (omega_bw * duration * omega0**2 * snr)
    uniform_sr = 1./(12.0 * sample_rate**2)
    if cr_bound >= uniform_delay:
        return uniform_delay, False
    else : 
        if cr_bound > threshold:
            return cr_bound, False
        else:
            return max(uniform_sr, cr_bound), True

def capped_barankin(delay, omega_bw, duration, snr, sample_rate, threshold = 1E-10):
        uniform_delay = delay **2/12.0 
        barankin_bound = 12.0 * np.pi /(omega_bw**3 * duration * snr)
        uniform_sr = 1./(12.0 * sample_rate**2)
        if barankin_bound >= uniform_delay:
            return uniform_delay, False
        else : 
            if barankin_bound > threshold:
                return barankin_bound, False
            else:
                return max(uniform_sr, barankin_bound), True

def whistle_crb_from_pair(metadata : AudioMetadata, snr, omega0, duration, delay, bandwidth : Optional[float] = None):
    """
    Calculate whistle Cramér-Rao Bound (CRB) from a pair of hydrophones.

    Parameters:
    - duration: Duration of the call.
    - hydrophone_pair: Pair of hydrophones.
    - frequency_range: Optional frequency range.

    Returns:
    - error estimate corresponding to Ziv Zakai bound
    - Mask indicating whether the pair should be used
    """
    if bandwidth is None :
        omega_bw = 300 * (2*np.pi) #100KHz bandwidth
    else : 
        omega_bw = bandwidth * 2 * np.pi
        
    if metadata.central_frequency <= 2500 :
        # Returns the capped CRB
        return capped_crb(delay, omega_bw, duration, omega0, snr, metadata.sample_rate)
    
    elif metadata.central_frequency <= 12500:
        # SNR Shifts at SNR3 and error jumps
        snr3 = snr3_value(omega_bw, duration, omega0)
        if snr >= snr3:
            return capped_crb(delay, omega_bw, duration, omega0, snr, metadata.sample_rate)
        else:
            return capped_barankin(delay, omega_bw, duration, snr, metadata.sample_rate)
        
    else:

        snr1 = snr1_value(omega_bw, duration, omega0, delay)

        if snr < snr1 :
            # As good as pure randomness
            return delay**2/12, False
        
        snr2 = snr2_value(omega_bw, duration, omega0)
        
        if snr < snr2:
            # Barankin bound 
            return capped_barankin(delay, omega_bw, duration, snr, metadata.sample_rate)
        
        snr3 = snr3_value(omega_bw, duration, omega0)

        if snr < snr3:
            # Transition mode
            return max(1./(12. * metadata.sample_rate**2), transition_bound(omega_bw, omega0, snr, delay, duration)), True
        
        else:
            # We reached CRB
            return capped_crb(delay,omega_bw, duration, omega0,snr,metadata.sample_rate)

def solve_delta(omega_bw, delay):
    y = (6./(omega_bw * delay))**2
    # Fonction f(x) = x*erf(x) - y
    def f(x):
        return x * phi(np.sqrt(x)) - y
    
    # Pour grandes x, x*erf(x) ~ x, donc la solution max ~ y
    # On cherche une racine au-dessus de y
    sol = root_scalar(f, bracket=[1, 50], method='bisect')   

    if not sol.converged:
        print("WARNING : Couldn't find the solution for delta")
        return 16.283008875250204

    return sol.root
    
def hfpc_crb_from_pair(metadata, snr, omega0, duration, delay, bandwidth : Optional[float] = None):
    """
    Calculate hfpc Cramér-Rao Bound (CRB) from a pair of hydrophones.

    Parameters:
    - duration: Duration of the call.
    - hydrophone_pair: Pair of hydrophones.
    - frequency_range: Optional frequency range.

    Returns:
    - error estimate corresponding to Ziv Zakai bound
    - Mask indicating whether the pair should be used
    """
    if bandwidth is None :
        omega_bw = 1E5 * (2*np.pi) #100KHz bandwidth
    else : 
        omega_bw = bandwidth * 2 * np.pi
    """
    ##### Thresholds #####
    gamma = 0.46 # Je pense pas besoin 
    if omega_bw == 1E5 * (2*np.pi) and delay == 0.000453125:
        delta = 16.283008875250204 #Solution de delta phi(racine(delta)) = (6/WD)^2, dans notre cas c'est une constante
    else :
        delta = solve_delta(omega_bw, delay)
    mu = (2.76/(np.pi**2))*(omega0/omega_bw)**2
    eta = (6./np.pi**2)*((omega0/omega_bw)**2)*(phi_inv((omega_bw**2/(24.*omega0**2)))**2)
    
    threshold_factor = omega_bw * duration / (2*np.pi) * snr # Duration pas très accurate
    print(omega_bw * duration / (2*np.pi))
    
    print(f'f0 : {omega0/6.28}')
    print(f'SNR : {snr}')
    print(f'Gamma : {gamma}\ndelta : {delta}\n mu : {mu}\n eta : {eta}')
    
    ##### Going through the curve ##### 
    uniform_error_bound = delay ** 2 /12.0
    if threshold_factor < gamma :
        return uniform_error_bound, False
    
    if threshold_factor < delta:
        barankin_delta = 6. / (omega_bw**2 * delta)
        linear_interpolation = (threshold_factor - gamma)/(delta-gamma) * (barankin_delta - uniform_error_bound) + uniform_error_bound
        return min(linear_interpolation, uniform_error_bound), False
    
    if threshold_factor < mu:
        return capped_barankin(delay, omega_bw, duration, snr, metadata.sample_rate)
    
    if threshold_factor < eta:
        barankin_mu = 6. / (omega_bw**2 * mu)
        crb_eta = 1./(2*omega0**2) * 1./eta
        linear_interpolation = (threshold_factor - mu)/(eta-mu) * (crb_eta - barankin_mu) + barankin_mu
        return min(max(linear_interpolation, 1./(12*metadata.sample_rate**2)) , uniform_error_bound), True
    
    else :
        return capped_crb(delay, omega_bw, duration, omega0, snr, metadata.sample_rate)
    """
    return capped_crb(delay, omega_bw, duration, omega0, snr, metadata.sample_rate)

def crb_from_pair(metadata : AudioMetadata, duration: float, hydrophone_pair: HydrophonePair , bandwidth : Optional[float] = None):
    """
    Calculate Cramér-Rao Bound (CRB) from a pair of hydrophones.

    Parameters:
    - call_type: Type of the call.
    - duration: Duration of the call.
    - hydrophone_pair: Pair of hydrophones.
    - frequency_range: Optional frequency range.

    Returns:
    - error estimate corresponding to Ziv Zakai bound
    - Mask indicating whether the pair should be used
    """
    ##### Computing intermediate variables #####
    snr_ref = hydrophone_pair.hydrophone_ref.snr_power
    snr_delta = hydrophone_pair.hydrophone_delta.snr_power
    best_error_bound = (1./metadata.sample_rate)**2 / 12
    if snr_delta is None or snr_ref is None or not(metadata.beluga_call_type in ['Whistle', 'HFPC']):
        return best_error_bound, True
    
    snr = snr_ref * snr_delta /(1+ snr_ref + snr_delta)
    omega0 = metadata.central_frequency * 2 * np.pi
    delay = 2.0 * hydrophone_pair.max_delay_idx / metadata.sample_rate

    if snr < 0 : 
            return delay**2/12., False
        
    ###### Main function #####
    
    if metadata.beluga_call_type == 'Whistle':
        return whistle_crb_from_pair(metadata, snr, omega0, duration, delay, bandwidth)
    
    if metadata.beluga_call_type == 'HFPC':
        return hfpc_crb_from_pair(metadata, snr, omega0, duration, delay, bandwidth)
    
    
    

def tdoas(audio_array: AudioArray, use_gcc : bool = False, compute_scores : bool = False):
    """
    Compute TDOAs from an audio array.

    Parameters:
    - audio_array: Filtered audio array.

    Returns:
    - tdoa_vector: TDOA vector with TDOAs at each pair.
    - crb_vector: CRB vector for error estimation at each pair.
    """
    tdoa_vector = []
    crb_vector = []
    tdoas_mask = []
    all_scores = [] if compute_scores else None
    
    for hydrophone_pair in audio_array.pairs_dict.values():
        ##### Computing #####
        tdoa, use_tdoa, scores = tdoa_from_pair(hydrophone_pair, audio_array.metadata.sample_rate, use_gcc, compute_scores)
        duration = audio_array.data_array.shape[1] / audio_array.metadata.sample_rate
        crb, use_crb = crb_from_pair(audio_array.metadata, duration, hydrophone_pair)
        
        ##### Agregating #####
        tdoa_vector.append(tdoa)
        crb_vector.append(crb)
        tdoas_mask.append(use_tdoa & use_crb)
        ###################################################################
        #tdoas_mask.append(True)
        ###################################################################
        if compute_scores and all_scores is not None:
            all_scores.append(scores)

    return np.array(tdoa_vector), np.array(crb_vector), np.array(tdoas_mask), all_scores

def tdoas_from_h1(audio_array: AudioArray, use_gcc : bool = False, compute_scores : bool = False):
    """
    Compute TDOAs from an audio array with h1 as a referential (ie output is of shape 3).
    Only for error evaluation purpose.

    Parameters:
    - audio_array: Filtered audio array.

    Returns:
    - tdoa_vector: TDOA vector with TDOAs at each pair.
    """
    tdoa_vector = []

    
    for i, (hydroname, hydrophone_pair) in enumerate(audio_array.pairs_dict.items()):
        
        if i <3:
            ##### Computing #####
            tdoa, use_tdoa, scores = tdoa_from_pair(hydrophone_pair, audio_array.metadata.sample_rate, use_gcc, compute_scores)
            
            tdoa_vector.append(tdoa)


    return np.array(tdoa_vector)