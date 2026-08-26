import numpy as np
# from librosa import frames_to_time, fft_frequencies 
# import os
from datetime import datetime
from pathlib import Path
import json
from pathlib import Path
from dataclasses import dataclass




# AUDIO parameters
SAMPLING_RATE = 384000
DURATION = 1
N = DURATION * SAMPLING_RATE 
MAX_TDOA = 1.1 * 0.3 / 1450
MAX_TDOA_IDX = int(SAMPLING_RATE*MAX_TDOA)
M = 4 # num microphones
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
CKPT_PATH = r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\phase_aware_tf_mask_260819_02.ckpt"

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

@dataclass(frozen=True, slots=True)
class ArrayParamters:
    num_mic = M 
    max_tdoa = MAX_TDOA

    def __post_init__(self) -> None:
        if self.num_mic <= 0:
            raise ValueError(f"num_mic must be greater than one, got {M}")

        if self.max_tdoa < 0:
            raise ValueError(f"max_tdoa must be positive, got {self.max_tdoa}")

@dataclass(frozen=True, slots=True)
class AudioParameters:
    sampling_rate = SAMPLING_RATE
    duration = DURATION
    min_freq = MIN_FREQ
    max_freq = MAX_FREQ

    def __post_init__(self) -> None:
        if self.sampling_rate < 0:
            raise ValueError("sampling_rate must be positive")

        if not 0 <= self.min_freq < self.max_freq <= self.sampling_rate / 2:
            raise ValueError("Invalid frequency range")

    @property
    def num_samples(self) -> int:
        return round(self.sampling_rate * self.duration)

@dataclass(frozen=True, slots=True)
class STFTParamters:
    window : str = 'hann'
    n_fft : int = N_FFT
    hop_length : int = HOP_LENGTH
    detrend : bool = False
    boundary : str | None = None
    padded : bool = False
    spectrogram_type : int = SPECTROGRAM_TYPE

@dataclass(frozen=True, slots=True)
class NetworkParameters:
    input_path : str
    image_size : int
    output_path : str
    checkpoint_path : str


@dataclass(frozen=True, slots=True)
class Parameters:
    audio : AudioParameters
    stft : STFTParamters
    array : ArrayParamters
    network : NetworkParameters

    @classmethod
    def from_json(cls, path: str | Path):
        path = Path(path)

        with path.open() as file: 
            data = json.load(file)

        network_data = data["network_parameters"].copy()
        network_data["checkpoint_path"] = Path(network_data["checkpoint_path"])

        return cls(
            audio=AudioParameters(**data["audio_parameters"]),
            stft=STFTParamters(**data["audio_parameters"]),
            array=ArrayParamters(**data["audio_parameters"]),
            network=NetworkParameters(**data["audio_parameters"]),
        )

    @property
    def stft_shape(self) -> tuple[int, int]:
        n, n_fft, hop_length, fs = self.audio.num_samples, self.stft.n_fft, self.stft.hop_length, self.audio.sampling_rate
        n_times =  int(1 + np.floor((n - n_fft) / hop_length))
        n_freqs = n_fft // 2 + 1
        n_freqs = int(np.floor(self.audio.max_freq * n_fft /fs) - np.ceil(self.audio.min_freq * n_fft / fs) + 1)
        return n_freqs, n_times

    