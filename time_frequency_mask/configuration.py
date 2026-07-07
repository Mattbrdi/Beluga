import numpy as np
from librosa import frames_to_time, fft_frequencies 
import os
from datetime import datetime
from pathlib import Path

# AUDIO parameters
SAMPLING_RATE = 384000
DURATION = 1
N = DURATION * SAMPLING_RATE 
MAX_TDOA = 0.3 / 1500

## STFT parameters
N_FFT = 4096
HOP_LENGTH = 2048
WINDOW_LIBROSA = "hann"

# Window parameters
MIN_FREQ = 500
MAX_FREQ = 20000
# Assuming no padding, and uncentered librosa stft
N_TIMES =  int(1 + np.floor((N - N_FFT) / HOP_LENGTH))
N_FREQS = N_FFT // 2 + 1
N_FREQS = int(np.floor(MAX_FREQ * N_FFT /SAMPLING_RATE) - np.ceil(MIN_FREQ * N_FFT / SAMPLING_RATE) + 1)
START_FREQ_IDX = int(np.ceil(MIN_FREQ * N_FFT / SAMPLING_RATE) + 1)

# Neural Network
IMAGE_SIZE = 256 

## Whistle BANK:
WHISTLE_WAV_PATH = r"C:\Users\amine\Desktop\Canada\Beluga\time_frequency_mask\data_generation\data\input\whistle2\wav"
WHISTLE_JSON_PATH = r"C:\Users\amine\Desktop\Canada\Beluga\time_frequency_mask\data_generation\data\input\whistle2\json"
WHISTLE_MASK_PATH = r"C:\Users\amine\Desktop\Canada\Beluga\time_frequency_mask\data_generation\data\input\whistle2\mask"
WHISTLE_PNG_PATH = r"C:\Users\amine\Desktop\Canada\Beluga\time_frequency_mask\data_generation\data\input\whistle2\png"

# Debug parameters
DEBUG_LEVEL = 0 

# Paths paremeters
DATASET_PATH = r"C:\Users\amine\Desktop\Canada\Beluga\time_frequency_mask\data_generation\data\output\260701_new_labeling"

now = datetime.now()
formatted_string = now.strftime("%Y%m%d%H%M%S") # Format to YYYYMMDDHHMMSS

formatted_string = "260701_new_labeling"

OUTPUT_PATH = Path(
    r"C:\Users\amine\Desktop\Canada\Beluga\time_frequency_mask\data_generation\data\output"
) / formatted_string

wav_dir = OUTPUT_PATH / "wav"
COUNT = sum(1 for p in wav_dir.iterdir()) if wav_dir.exists() else 0

#Spectrogram type defines how is scaled the spectrogram output when plotted
# 0 : max is max
# 1 : scaled by 99th percentile
# 2: scaled by 95th percentile

SPECTROGRAM_TYPE = 1

if SPECTROGRAM_TYPE not in [0, 1, 2]:
    raise ValueError(f"Incorrect spectrogram_type parameter got {SPECTROGRAM_TYPE} not in [0, 1, 2]")


