"""Implementation of the VMD algorithm from Dragomiretskiy and Zosso."""

import numpy as np
from scipy.fft import fft, fftfreq, ifft
from src.utils.sub_classes import VMDOptions

def converged_bool(new_u_hat: np.ndarray, old_u_hat :np.ndarray, vmd_options:VMDOptions, total_iter:int, mask:np.ndarray):
    """Check if the VMD converged or reached max iterations

    Args:
        new_u_hat (np.ndarray): Values of u in Fourrier domain at new iteration
        old_u_hat (np.ndarray): Former values
        vmd_options (VMDOptions): Options for the algorithm
        total_iter (int): Maximum amount of iterations
        mask (np.ndarray): Only compute for positive frequencies

    Returns:
        bool: True if the algorithm should stop
    """
    if total_iter >= vmd_options.max_iter:
        return True
    return np.sum((np.linalg.norm(new_u_hat[:,mask] - old_u_hat[:,mask], axis = 1)/(np.linalg.norm(old_u_hat[:,mask],axis=1)+1E-34))**2) < vmd_options.tolerance

def vmd_decomposition(mono_canal, vmd_options:VMDOptions, fech, omega_k = None):
    # Audio array contient le call type
    f_hat = fft(mono_canal).reshape(1,-1) # shape 1,N
    frequencies = fftfreq(len(mono_canal), d=1.) # shape 1 , N
    # TODO : d = 1./frequencies ?? Tester les valeurs de frequencies
    pos_freqs = frequencies[frequencies >= 0]
    mask = (frequencies >= 0)
    N = len(frequencies)
    frequencies = frequencies.reshape(1, -1)
    if omega_k is None:
        omega_k = np.linspace(0,1./(2*np.pi),vmd_options.nb_modes).reshape(-1,1) # Omegas between 0 and Nyquist frequency
    print(f"OG Omegas : {omega_k}")
    old_u_hat = np.zeros((vmd_options.nb_modes,N), dtype=complex) # shape K, N
    new_u_hat = old_u_hat # Just to clarify code
    lambdas = np.zeros((1,N), dtype = complex) # Shape 1,N
    total_iter = 0
    converged = False
    const_term = f_hat + lambdas / 2
    old_u_squared_sum = 1E-34
    while not(converged):
        # Compute the norm of uk^n+1 - uk ^n to avoid copy
        norm_of_difference = np.zeros(vmd_options.nb_modes)
        # Update uk
        denom = (1 + 2*vmd_options.alpha*(frequencies - omega_k)**2) # (K, N)
        sum_lower_k = 0
        sum_upper_k = np.sum(old_u_hat[1:],axis=0)
        new_val = (const_term - sum_upper_k) / denom[0]
        norm_of_difference[0] = np.linalg.norm(new_val[0,mask]-old_u_hat[0,mask])**2
        new_u_hat[0] = new_val
        for k in range(1,vmd_options.nb_modes):
            sum_lower_k += new_u_hat[k-1]
            sum_upper_k -= old_u_hat[k]
            new_val = (const_term - sum_lower_k - sum_upper_k) / denom[k]
            norm_of_difference[k] = np.linalg.norm(new_val[0,mask]-old_u_hat[k,mask])**2
            new_u_hat[k] = new_val
        # Update omegas
        u_squared = np.abs(new_u_hat[:,mask])**2
        upper_integral = np.trapz(pos_freqs * u_squared, x = pos_freqs, axis = 1) # Should be of shape (K,1)
        lower_integral = np.trapz(u_squared, x = pos_freqs, axis = 1)
        omega_k = (upper_integral/lower_integral).reshape(-1,1) # Should be of shape (K,1)
        if vmd_options.tau != 0.:
            lambdas += vmd_options.tau * (f_hat - np.sum(new_u_hat, axis = 0, keepdims = True))
            const_term = f_hat + lambdas / 2
        total_iter += 1
        converged = np.sum(norm_of_difference/old_u_squared_sum) < vmd_options.tolerance
        converged = np.logical_or(converged,total_iter >= vmd_options.max_iter)
        if not converged:
            old_u_squared_sum = np.sum(u_squared,axis=1) +1E-34
    print(f'Total iteration : {total_iter}')
    print(f"Final Omegas : {omega_k}")
    u = np.real(ifft(new_u_hat, axis = 1))
    return u, new_u_hat, omega_k
        