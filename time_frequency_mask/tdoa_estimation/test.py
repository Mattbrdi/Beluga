
import numpy as np 
import sys
from pathlib import Path

from scipy.signal import correlate
from scipy.fft import fft, ifft

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BSS.Utils.signal_class import Signal, MultiSignal, Mixture

from time_frequency_mask.configuration import DURATION, SAMPLING_RATE, MAX_TDOA
from time_frequency_mask.plotter import plot_spectrogram_4D, plot_waveform_4D, plot_mask
from time_frequency_mask.tdoa_estimation.blob import Blob, output_blobs_from_mask, blob_filtering_heuristic
from time_frequency_mask.tdoa_estimation.tdoa import compute_tdoa, compute_cross_corr
from time_frequency_mask.data_generation.models.mask import AudioMask
from time_frequency_mask.data_generation.io.data_parser import read_wav_file
from time_frequency_mask.data_generation.core.preprocess import bandpass_filter
from time_frequency_mask.masknet.run_inference import get_mask_from_array

WAV_PATH = r"C:\Users\amine\Downloads\amine.wav"#r"C:\Users\amine\Desktop\Canada\Beluga\time_frequency_mask\data_generation\data\input\beluga_2026_7.wav"

synthetic = True
plot = True
denoise = False 
std_noise = 5
amplitude = 1
# n_tdoa = int(np.ceil(SAMPLING_RATE*MAX_TDOA))
n_tdoa = int(SAMPLING_RATE*MAX_TDOA)

delay_matrix = np.array([
    [0],  
    [59],
    [58],
    [10]])

def get_synthetic_signal(duration = DURATION, fs = SAMPLING_RATE, amplitude = amplitude):
    s1 = Signal.generate_multi_freq_signal(
        duration, fs, start_time=0.15, end_time=0.45, 
        frequencies=[5000,2000], window_type='hann'
    )
    s2 = Signal.generate_multi_freq_signal(
        duration, fs, start_time=0.35, end_time=0.75, 
        frequencies=[5100,7000], window_type='hann'
    )
    # s1 = (s1 + 3*s2)*amplitude

    signal_energy = np.sum(s1.data*s1.data)

    msignal = MultiSignal([s1])
    return msignal, signal_energy

def get_signal_from_path(path = WAV_PATH):
    waveform, frame_rate = read_wav_file(WAV_PATH)
    msignal = MultiSignal.from_array(waveform[np.newaxis, 0], frame_rate)
    return msignal

def apply_delay(delay_matrix, msignal):
    mixer = Mixture.create_delay_mixture(E=1, S=4, L=20000, delay_matrix=delay_matrix)
    tetra = mixer.apply(msignal, mode = "same") # dans tetra.data matrice (micro, sample)
    return tetra

def apply_noise(tetra, std_noise = std_noise):
    
    rng = np.random.default_rng(seed=42)  
    noise = MultiSignal.from_array(data = rng.normal(loc=0, scale=std_noise, size = tetra.data.shape), fs = SAMPLING_RATE)
    return tetra + noise

def main():
    msignal = None
    signal_energy = None
    if synthetic:
        msignal, signal_energy = get_synthetic_signal()
    else:
        msignal = get_signal_from_path()

    tetra = apply_delay(delay_matrix, msignal)

    tetra_noised = apply_noise(tetra).data

    if denoise:
        from scipy.signal import wiener

        noise_std = std_noise  # known std of the added noise
        noise_var = noise_std ** 2

        # Wiener filter; mysize controls the local window size
        tetra_noised = wiener(tetra_noised, mysize=(1, 1000), noise=noise_var)

    mask = AudioMask(get_mask_from_array(tetra_noised))

    if signal_energy is not None:
        active_count = np.count_nonzero(msignal.data[0])
        SNR = 10 * np.log10(signal_energy / (active_count * std_noise**2))
        print(f"snr estimation: {SNR}")
    # mask.data = np.zeros_like(mask.data)
    
    # mask.data[48:49,60:-60] = True

    if plot:
        plot_waveform_4D(tetra_noised, SAMPLING_RATE)
        # plot_mask(mask.data)
        plot_spectrogram_4D(tetra_noised, SAMPLING_RATE, is_db=False, mask = mask.data)

    blobs = output_blobs_from_mask(mask)

    blobs = blob_filtering_heuristic(blobs)

    for i, blob in enumerate(blobs):
        tmin_idx = blob.tmin_idx
        tmax_idx = blob.tmax_idx

        idx_center = (tmin_idx + tmax_idx) // 2
        win_size = tmax_idx - tmin_idx

        fmin = blob.fmin
        fmax = blob.fmax

        canal = np.copy(tetra_noised)

        canal = bandpass_filter(canal, SAMPLING_RATE, fmin, fmax)
        print(f"tdoa estimation for blob {i} is:")

        tdoa_01 = compute_tdoa(canal[0], canal[1], idx_center, win_size)
        tdoa_02 = compute_tdoa(canal[0], canal[2], idx_center, win_size)
        tdoa_03 = compute_tdoa(canal[0], canal[3], idx_center, win_size)

        print(f"tdoa are : tdoa_01 {tdoa_01}, tdoa_02 {tdoa_02}, tdoa_03 {tdoa_03}")

    correlations_01 = []
    correlations_02 = []
    correlations_03 = []
    for i, blob in enumerate(blobs):
        tmin_idx = blob.tmin_idx
        tmax_idx = blob.tmax_idx

        idx_center = (tmin_idx + tmax_idx) // 2
        win_size = tmax_idx - tmin_idx

        fmin = blob.fmin
        fmax = blob.fmax

        canal = np.copy(tetra_noised)

        canal = bandpass_filter(canal, SAMPLING_RATE, fmin, fmax)

        correlations_01.append(blob.area*compute_cross_corr(canal[0], canal[1], idx_center, win_size))
        correlations_02.append(blob.area*compute_cross_corr(canal[0], canal[2], idx_center, win_size))
        correlations_03.append(blob.area*compute_cross_corr(canal[0], canal[3], idx_center, win_size))
        # blob.area*np.sqrt(win_size*(fmax-fmin))


    print(f"tdoa estimation for merged blob estimator is:")
    tdoa_prime_01 = -(np.argmax(np.sum(correlations_01, axis=0)) - n_tdoa)
    tdoa_prime_02 = -(np.argmax(np.sum(correlations_02, axis=0)) - n_tdoa)
    tdoa_prime_03 = -(np.argmax(np.sum(correlations_03, axis=0)) - n_tdoa)

    print(f"tdoa are : tdoa_01 {tdoa_prime_01}, tdoa_02 {tdoa_prime_02}, tdoa_03 {tdoa_prime_03}")


    print("Normal tdoa estimation:")
    waveform_crosscorr_01 = correlate(tetra_noised[0], tetra_noised[1])[len(tetra_noised[0]) - 76 : len(tetra_noised[0]) + 76]
    waveform_crosscorr_02 = correlate(tetra_noised[0], tetra_noised[2])[len(tetra_noised[0]) - 76 : len(tetra_noised[0]) + 76]
    waveform_crosscorr_03 = correlate(tetra_noised[0], tetra_noised[3])[len(tetra_noised[0]) - 76 : len(tetra_noised[0]) + 76]
    print(-np.argmax(waveform_crosscorr_01) + 76 - 1)
    print(-np.argmax(waveform_crosscorr_02) + 76 - 1)
    print(-np.argmax(waveform_crosscorr_03) + 76 - 1)

    def gcc_phat_from_pair(audio_one, audio_two):

        if len(audio_one) != len(audio_two):
            raise ValueError("Audio signals must have the same length for GCC-PHAT.")
        
        fft_one = fft(audio_one, 2*len(audio_one) - 1)
        fft_two = fft(audio_two, 2*len(audio_one) - 1)
        
        cross_corr_spectrum = fft_one * np.conj(fft_two)
        phat_function = 1./np.abs(cross_corr_spectrum)
        gcc_phat = ifft(cross_corr_spectrum * phat_function).real
        
        return np.roll(gcc_phat, len(audio_one) - 1)  # Shift to center the zero lag
    
    center = len(tetra_noised[0]) - 1

    gcc_01 = gcc_phat_from_pair(tetra_noised[0], tetra_noised[1])[center - n_tdoa : center + n_tdoa + 1]
    gcc_02 = gcc_phat_from_pair(tetra_noised[0], tetra_noised[2])[center - n_tdoa : center + n_tdoa + 1]
    gcc_03 = gcc_phat_from_pair(tetra_noised[0], tetra_noised[3])[center - n_tdoa : center + n_tdoa + 1]

    print("GCC-PHAT tdoa estimation:")
    print(-np.argmax(gcc_01) + n_tdoa)
    print(-np.argmax(gcc_02) + n_tdoa)
    print(-np.argmax(gcc_03) + n_tdoa)

if __name__ == "__main__":
    main()
