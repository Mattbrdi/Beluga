
import numpy as np 

def SNR_in(signal, noise):
    """ Computes the input SNR

    @param signal The input signal to be denoised
    @param noise The noise signal 

    @return SNR_in
    """
    return 10 * np.log10(np.dot(signal, signal) / np.dot(noise, noise))

def SNR_out(signal, denoised_signal):
    """ Computes the output SNR

    @param signal The input signal to be denoised
    @param denoised_signal The input signal after denoising 

    @return SNR_out
    """
    return 10 * np.log10(np.dot(signal, signal) / np.dot(signal - denoised_signal, signal - denoised_signal))

def NMRSE(signal, denoised_signal):
    """ Computes the normalized root mean square error which is used to measure the difference between the denoised and reference signals

    @param signal The input signal to be denoised
    @param denoised_signal The input signal after denoising 

    @return NMRSE
    """
    n = len(signal)
    return np.sqrt( (1 / n) * np.dot(signal - denoised_signal, signal - denoised_signal)) / (np.max(signal) - np.min(signal))

def PCC(signal, denoised_signal):
    """ Computes the Pearson correlation coefficient (PCC) which assesses the degree of linear association

    @param signal The input signal to be denoised
    @param denoised_signal The input signal after denoising 

    @return Pearson correlation coefficient
    """
    sigma_s = np.std(signal)
    sigma_ds = np.std(denoised_signal)
    return np.cov(signal, denoised_signal) / (sigma_s * sigma_ds)
