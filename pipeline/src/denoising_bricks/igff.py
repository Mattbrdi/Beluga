"""Module to filter IMFs using IGFF algorithm
This module contains the functions to filter IMFs using the IGFF algorithm.
"""
import numpy as np
from scipy.stats import norm
from scipy.special import gamma, psi

from collections import defaultdict


class FRCMGDEOptions:
    """Options for the FRCMGDE algorithm depending on the call type."""
    def __init__(self, calltype: str):
        """
        Initialise les paramètres selon le type d'appel.
        Args:
            calltype (str): Type d'appel ('ec', 'cc', 'whistle', 'hfpc')
        """
        if calltype == 'ec' or calltype == 'cc':
            self.tau = 3  # Scale factor (all values between 1 and tau are considered)
            self.p = 30  # grids
            self.I = 2  # Embedding dimension
            self.d = 2  # Delay for Takens reconstructions
            self.alpha = 0.8  # Fractional derivative in [0,1]. We raise alpha to detect rare events like beluga clicks
        elif calltype == 'whistle' or calltype == 'hfpc':
            self.tau = 10 
            self.p = 10
            self.I = 4
            self.d = 6
            self.alpha = 0.4
        self.calltype = calltype
   
def pk(audio_array):
    """Calcule le peak-to-peak d'un tableau audio.
    Args:
        audio_array (np.ndarray): Signal audio.
    Returns:
        float: Valeur peak-to-peak.
    """
    return(np.max(audio_array) - np.min(audio_array))

def std(audio_array):
    """Calcule l'écart-type d'un tableau audio.
    Args:
        audio_array (np.ndarray): Signal audio.
    Returns:
        float: Écart-type.
    """
    return(np.std(audio_array, ddof = 1))

def rms(audio_array):
    """Calcule la valeur RMS d'un tableau audio.
    Args:
        audio_array (np.ndarray): Signal audio.
    Returns:
        float: Valeur RMS.
    """
    return(np.sqrt(np.mean(audio_array**2)))

def rav(audio_array):
    """Calcule la valeur absolue moyenne d'un tableau audio.
    Args:
        audio_array (np.ndarray): Signal audio.
    Returns:
        float: Valeur absolue moyenne.
    """
    return(np.mean(np.abs(audio_array)))

def coarse_grained_values(audio_array, tau):
    """Refined Composite Multiscale Step.
    Args:
        audio_array (np.ndarray): 1D numpy array.
        tau (int): Scale factor.
    Returns:
        np.ndarray: Tableau de la série temporelle coarse-grained.
    """
    N = len(audio_array)
    # Calculate the number of coarse-grained segments
    M = N // tau
    # Reshape the array and sum over the segments
    coarsed_array = audio_array[:M*tau].reshape(-1, tau).sum(axis=1)
    # Normalize by the scale factor tau
    return coarsed_array / tau

def coarsed_distribution(coarsed_array):
    """
    Maps the values of the coarsed array between 0 and 1.
    Args:
        coarsed_array (np.ndarray): 1D numpy array.
    Returns:
        np.ndarray: Distribution extraite de la série.
    """
    mu = np.mean(coarsed_array)
    sigma = np.std(coarsed_array, ddof=1)
    return norm.cdf(coarsed_array, loc = mu, scale = sigma)

def integer_mapping(mapped_array, p, I, d):
    """
    Integer mapping of the values.
    Args:
        mapped_array (np.ndarray): 1D numpy array.
        p (int): Number of grids.
        I (int): Embedding dimension.
        d (int): Delay for Takens reconstruction.
    Returns:
        tuple: Dispersion modes et leurs fréquences.
    """
    integer_array = np.floor(p*mapped_array + 0.5).astype(int) # P^p_j
    M = len(integer_array)
    N = M - (I-1) * d
    indexes = np.arange(N)[:, None] + np.arange(I) * d
    embedded_matrix  = integer_array[indexes]  # P^{I,p}_i, shape (N, I)
    modes, frequencies = np.unique(embedded_matrix, axis=0, return_counts=True)
    frequencies = frequencies.astype(float) / N
    return modes, frequencies

def frcmgde(audio_array, frcm_options : FRCMGDEOptions):
    """
    FRCMGDE algorithm.
    Args:
        audio_array (np.ndarray): 1D numpy array.
        frcm_options (FRCMGDEOptions): Options pour l'algorithme.
    Returns:
        float: Valeur FRCMGDE du signal audio.
    """
    accumulate = defaultdict(float)
    for tau in range(1, frcm_options.tau+1):
        coarsed_array = coarse_grained_values(audio_array, tau)
        mapped_array = coarsed_distribution(coarsed_array)
        modes, frequencies = integer_mapping(mapped_array, frcm_options.p, frcm_options.I, frcm_options.d)
        for mode, frequency in zip(modes, frequencies):
            key = tuple(mode)
            accumulate[key] += frequency
    summed_freqs = 1./frcm_options.tau * np.array(list(accumulate.values())) # Value of f bar
    frcmgde = -1./gamma(1 + frcm_options.alpha)*np.sum(summed_freqs**(1-frcm_options.alpha) * (np.log(summed_freqs)+ psi(1) - psi(1-frcm_options.alpha)))
    return frcmgde

def feature_matrix(imfs, calltype:str):
    """
    Calcule la matrice de features pour une liste d'IMFs.
    Args:
        imfs (list[np.ndarray]): Liste des IMFs.
        calltype (str): Type d'appel.
    Returns:
        np.ndarray: Matrice de features.
    """
    options = FRCMGDEOptions(calltype)
    vector = []
    for imf in imfs:
        rms_ = rms(imf)
        rav_ = rav(imf)
        frcmgde_ = frcmgde(imf, options)
        if calltype == 'ec' or calltype == 'cc':
            pk_ = pk(imf)
            vector += [pk_,rms_,rav_, frcmgde_]
        elif calltype == 'whistle' or calltype == 'hfpc':
            std_ = std(imf)
            vector += [std_,rms_, rav_, frcmgde_]
    feature_mat = np.array(vector).reshape(-1,4)
    return feature_mat

def igff(feature_matrix):
    """
    IGFF algorithm for mode discrimination
    Args:
        feature_matrix (np.ndarray): Matrice de features.
    Returns:
        np.ndarray: Valeurs IGFF pour chaque IMF.
    """
    #All features are negative features
    normalized_feature_matrix = 0.998 *(np.max(feature_matrix, axis=0) - feature_matrix)/(np.max(feature_matrix, axis=0)-np.min(feature_matrix, axis=0)) +0.002
    gravity_matrix = normalized_feature_matrix / np.sum(normalized_feature_matrix, axis=0) # yij = xij / sum_i(xij)
    redundancy_vector = 1-(-1./np.log(np.shape(feature_matrix)[0]) * np.sum(gravity_matrix*np.log(gravity_matrix), axis = 0)) # yij log(yij), shape(imfs)[0] is n, the number of imfs, should be of size 5
    weights = redundancy_vector / np.sum(redundancy_vector)
    return np.dot(weights, feature_matrix.T)

def split_modes(imfs,computed_igff, threshold):
    """
    Sépare les IMFs en purs et bruités selon le seuil IGFF.
    Args:
        imfs (np.ndarray): IMFs.
        computed_igff (np.ndarray): Valeurs IGFF.
        threshold (float): Seuil.
    Returns:
        tuple: (pure_imfs, noisy_imfs)
    """
    pure_imfs = imfs[computed_igff > threshold]
    noisy_imfs = imfs[computed_igff <= threshold]
    return pure_imfs, noisy_imfs

def imf_filter(imfs, call_type:str, imf_threshold):
    """
    Calcule IGFF et filtre les IMFs.
    Args:
        imfs (np.ndarray): IMFs.
        call_type (str): Type d'appel.
        imf_threshold (float): Seuil IGFF.
    Returns:
        tuple: (pure_imfs, noisy_imfs, computed_igff)
    """
    frcmgde_options = FRCMGDEOptions(call_type)
    feature_mat = feature_matrix(imfs, frcmgde_options.calltype)
    computed_igff = igff(feature_mat)
    max_igff = np.max(computed_igff)
    min_igff = np.min(computed_igff)
    computed_igff = (computed_igff - min_igff)/(max_igff-min_igff)
    pure_imfs, noisy_imfs = split_modes(imfs,computed_igff,imf_threshold)
    return pure_imfs, noisy_imfs , computed_igff



    
    
    