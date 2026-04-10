"""Implementation of WOA algorithm for optimal VMD parameters."""

import numpy as np  
from copy import deepcopy
from src.denoising_bricks.vmd import vmd_decomposition
from src.utils.sub_classes import VMDOptions

def gen_whale_coeffs(total_iter, max_iter):
    """
    Generate coefficients for the Whale Optimization Algorithm.

    Parameters:
    - total_iter: Current iteration number.
    - max_iter: Maximum number of iterations.

    Returns:
    - l: Random value between -1 and 1.
    - p: Random value between 0 and 1.
    - a_vector: Vector for position update.
    - c_vector: Vector for position update.
    """
    l = 2*np.random.rand()-1 # l in (-1,1)
    p = np.random.rand() # p in (0,1)
    a = 2 - (2 * total_iter / max_iter)
    r = np.random.rand(2)
    a_vector = 2 * a * r - a
    c_vector = 2 * r
    return(l, p, a_vector, c_vector)

def window(value, min_bound, max_bound):
    """
    Constrain a value within specified bounds.

    Parameters:
    - value: Value to be constrained.
    - min_bound: Minimum bound.
    - max_bound: Maximum bound.

    Returns:
    - value: Constrained value.
    """
    if value > max_bound :
        value = max_bound
    elif value < min_bound:
        value = min_bound
    return value

class Whale:
    """
    Represents a whale in the Whale Optimization Algorithm.
    """
    def __init__(self, parameters):
        """
        Initialize a whale with random K and alpha values.

        Parameters:
        - parameters: Parameters for VMD denoising.
        """
        self.K = np.random.randint(parameters.woa_options.min_k_bound, parameters.woa_options.max_k_bound)
        self.alpha = np.random.randint(parameters.woa_options.min_alpha_bound, parameters.woa_options.max_alpha_bound)
        self.pse = np.inf
        self.vmd_options = VMDOptions(
            parameters.vmd_options.max_iter,
            self.K,
            parameters.vmd_options.tolerance,
            self.alpha,
            parameters.vmd_options.tau
            )

    def fitness(self, audio_array, fech):
        """
        Calculate the fitness of the whale based on VMD decomposition.

        Parameters:
        - audio_array: Input audio array.
        - fech: Sampling frequency.

        Returns:
        - pse: Power Spectrum Entropy.
        """
        # Run VMD
        self.vmd_options.set_k_alpha(self.K,self.alpha)
        _, u_hat, _ = vmd_decomposition(audio_array, self.vmd_options, fech)
        pse = 0
        power_spectrum = 1./(2*np.pi * np.shape(u_hat)[0])*np.abs(u_hat)**2 # Table where one row = one mode
        density = power_spectrum / np.sum(power_spectrum, axis = 1, keepdims=True)
        pse -= np.sum(density * np.log(density))
        self.pse = pse
        return pse

def woa_algorithm(audio_array : np.ndarray, vmd_parameters, fech):
    """
    Whale Optimization Algorithm for VMD denoising.

    Parameters:
    - audio_array: Input audio array.
    - vmd_parameters: Parameters for VMD denoising.
    - fech: Sampling frequency.

    Returns:
    - best_K: Best K value found.
    - best_alpha: Best alpha value found.
    - best_pse: Best Power Spectrum Entropy found.
    """
    whales = [Whale(vmd_parameters) for _ in range(vmd_parameters.woa_options.pop_size)]
    total_iter = 0
    best_K = np.random.randint(vmd_parameters.woa_options.min_k_bound, vmd_parameters.woa_options.max_k_bound)
    best_alpha = np.random.randint(vmd_parameters.woa_options.min_alpha_bound, vmd_parameters.woa_options.max_alpha_bound)
    best_pse = np.inf
    while total_iter < vmd_parameters.woa_options.max_iter:
        #print(best_K, best_alpha, best_pse)
        old_whales = deepcopy(whales)
        for whale in whales:
            l, p, a_vector, c_vector = gen_whale_coeffs(total_iter, vmd_parameters.woa_options.max_iter)
            if p<0.5:
                if np.linalg.norm(a_vector)<1:
                    d_vector = np.abs([c_vector[0]*best_K-whale.K,c_vector[1]*best_alpha-whale.alpha])
                    new_K = best_K - a_vector[0]*d_vector[0]
                    new_alpha = best_alpha - a_vector[1]*d_vector[1]
                else :
                    other_whale = np.random.choice(old_whales)
                    d_vector = np.abs([c_vector[0]*other_whale.K-whale.K,c_vector[1]*other_whale.alpha-whale.alpha])
                    new_K = other_whale.K - a_vector[0]*d_vector[0]
                    new_alpha = other_whale.alpha - a_vector[1]*d_vector[1]
            else :
                d_vector = np.abs([best_K-whale.K,best_alpha-whale.alpha])
                # We set b = 1
                new_K = d_vector[0]*np.exp(l)*np.cos(2*np.pi*l) + best_K
                new_alpha = d_vector[1]*np.exp(l)*np.cos(2*np.pi*l) + best_alpha
            new_K = window(new_K, vmd_parameters.woa_options.min_k_bound, vmd_parameters.woa_options.max_k_bound)
            new_alpha = window(new_alpha, vmd_parameters.woa_options.min_alpha_bound, vmd_parameters.woa_options.max_alpha_bound)
            whale.K = new_K
            whale.alpha = new_alpha
        for whale in whales:
            new_pse = whale.fitness(audio_array, fech)
            if new_pse < best_pse:
                best_K = whale.K
                best_alpha = whale.alpha
                best_pse = new_pse
        total_iter +=1
    return int(best_K), int(best_alpha), best_pse