# code taken from https://github.com/gdetor/wavelet_denoising/blob/master/example.py

import numpy as np
# import pandas as pd
import matplotlib.pylab as plt

# from scipy.signal import butter, filtfilt
from scipy.signal import spectrogram

from denoising import WaveletDenoising

from src.mathematics.metrics import SNR_in, SNR_out

def plot_coeffs_distribution(coeffs):
    """! Plots all the wavelet decomposition's coefficients. """
    fig = plt.figure()
    size_ = int(len(coeffs) // 2) + 1
    if size_ % 2 != 0:
        size_ = size_+1

    for i in range(len(coeffs)):
        ax = fig.add_subplot(size_, 2, i+1)
        ax.hist(coeffs[i], bins=50)


def pretty_plot(data, titles, palet, fs=1, length=100, nperseg=256):
    """! Plots the contents of the list data. """
    fig = plt.figure(figsize=(13, 13))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    index = 1
    for i, d in enumerate(data):
        ax = fig.add_subplot(8, 2, index)
        ax.plot(d[:length], color=palet[i])
        ax.set_title(titles[i])
        ax = fig.add_subplot(8, 2, index+1)
        f, t, Sxx = spectrogram(d, fs=fs, nperseg=nperseg)
        ax.pcolormesh(t, f, Sxx, shading='auto')
        index += 2


def run_experiment(data, level=2, fs=1, f0 = 1, nperseg=256, length=100):
    """! Run the wavelet denoising over the input data for each threshold
    method.
    """

    # Experiments titles / thresholding methods
    titles = ['Original data',
              'Universal Method',
              'SURE Method',
              'Energy Method',
              'SQTWOLOG Method',
              'Heursure Method',
              'SI-ACF Method']

    # Theshold methods
    experiment = ['universal',
                  'stein',
                  'energy',
                  'sqtwolog',
                  'heurstein',
                  "si_acf"]

    # WaveletDenoising class instance
    wd = WaveletDenoising(normalize=False,
                          wavelet='db3',
                          transform='swt',
                          level=level,
                          thr_mode='soft',
                          selected_level=level,
                          method="universal",
                          energy_perc=0.90,
                          fs=fs,
                          ff=f0)

    # Run all the experiments, first element in res is the original data
    res = [data]
    for i, e in enumerate(experiment):
        wd.method = experiment[i]
        res.append(wd.fit(data))

    # Plot all the results for comparison
    palet = ['r', 'b', 'k', 'm', 'c', 'orange', 'g', 'y']
    pretty_plot(res,
                titles,
                palet,
                fs=fs,
                length=length,
                nperseg=nperseg)
    
    return res


if __name__ == '__main__':
    # ECG Data
    import pandas as pd
    # fs = 20
    # raw_data = pd.read_pickle("./data/apnea_ecg.pkl")
    # N = int(len(raw_data) // 1000)
    # N = min(2048, N)
    # data = raw_data[:N].values
    # data = data[:, 0]

    fs = 44100  # sample rate
    duration = 1.5  # seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # Base frequency (center of whistle)
    f0 = 1000  # Hz

    # Frequency modulation (slow + fast wobble)
    f_mod = 800 * np.sin(2 * np.pi * 2 * t)      # slow sweep
    f_vibrato = 200 * np.sin(2 * np.pi * 8 * t)  # faster vibrato

    # Instantaneous frequency
    f_t = f0 + f_mod + f_vibrato

    # Integrate frequency to phase
    phase = 2 * np.pi * np.cumsum(f_t) / fs

    # Generate waveform
    signal = np.sin(phase)

    # Apply amplitude envelope (fade in/out)
    envelope = np.exp(-3 * t) * (1 - np.exp(-10 * t))
    signal *= envelope

    # Normalize
    signal /= np.max(np.abs(signal))

    print(signal.shape)  # NumPy array

    data = signal[:65536]
    print(np.size(data))


    noise = np.random.rand(np.shape(data)[0])
    data = data + noise / 20
    res = run_experiment(data, level=4, fs=fs, f0 = f0)

    # EEG Data
    # raw_data = np.genfromtxt("./data/Z001.txt")
    # fc = 40
    # fs = 173.61
    # w = fc / (fs / 2)
    # b, a = butter(5, w, 'low')
    # data = filtfilt(b, a, raw_data)
    # run_experiment(data, level=4, fs=fs)
    plt.show()

    print("SNR_in", SNR_in(data, noise))
    for r in res[1:]:
        print("SNR_out", SNR_out(data + noise, r))