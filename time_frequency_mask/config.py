import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True, slots=True)
class ArrayParameters:
    num_mics : int 
    max_tdoa : float
    array_geometry : np.ndarray
    sound_speed : float

    def __post_init__(self):
        object.__setattr__(self, 'array_geometry', np.asarray(self.array_geometry))

        if self.num_mics <= 1:
            raise ValueError(f"num_mics must be greater than one, got {self.num_mics}")

        if self.num_mics != len(self.array_geometry):
            raise ValueError(f"num_mics and array_geometry does not match, got {self.num_mics} mics "
                             f"and {len(self.array_geometry)} in the array geometry.")

        if self.max_tdoa < 0:
            raise ValueError(f"max_tdoa must be positive, got {self.max_tdoa}")

        if self.sound_speed < 0:
            raise ValueError(f"sound_speed must be positive, got {self.sound_speed}")

@dataclass(frozen=True, slots=True)
class AudioParameters:
    sampling_rate : int
    duration : float
    min_duration : float
    max_duration : float
    min_freq : float
    max_freq : float

    def __post_init__(self) -> None:
        if self.sampling_rate < 0:
            raise ValueError("sampling_rate must be positive")

        if not 0 <= self.min_freq < self.max_freq <= self.sampling_rate / 2:
            raise ValueError("Invalid frequency range")

        if not 0 <= self.min_duration <= self.max_duration:
            raise ValueError("Invalid duration range")

    @property
    def num_samples(self) -> int:
        if self.duration is None:
            raise ValueError("No default num_samples when audioparameters.duration is set to None. "
                             "Please provide a value in the json file")

        return int(self.sampling_rate * self.duration)

@dataclass(frozen=True, slots=True)
class STFTParamters:
    n_fft : int
    hop_length : int
    window : str = 'hann'
    detrend : bool = False
    boundary : str | None = None
    padded : bool = False
    spectrogram_type : int = 0

    def __post_init__(self):
        if self.spectrogram_type not in (0, 1, 2):
            raise ValueError("Incorrect spectrom_type")

    def num_time_bins(self, num_samples: int) -> int:
        if num_samples < self.n_fft:
            raise ValueError(
                f"Input has {num_samples} samples, but n_fft={self.n_fft}"
            )

        if self.boundary is not None or self.padded:
            raise NotImplementedError(
                "Time-bin calculation currently supports "
                "boundary=None and padded=False"
            )

        return 1 + np.floor((num_samples - self.n_fft) // self.hop_length)

    def freq_index(self, sampling_rate : float, freq : float) -> int:
        return int(np.ceil(freq * self.n_fft / sampling_rate) + 1)


    def num_frequency_bins(self) -> int:
        return self.n_fft // 2 + 1

    def num_frequency_bins_between(
        self,
        sampling_rate : int,
        min_freq : float,
        max_freq : float,
    ) -> int:
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        
        if not 0 <= min_freq < max_freq <= sampling_rate /2:
            raise ValueError(f"Incorrect frequency range")

        return int(np.floor(max_freq * self.n_fft /sampling_rate) - np.ceil(min_freq * self.n_fft / sampling_rate) + 1)

@dataclass(frozen=True, slots=True)
class NetworkParameters:
    input_path : Path
    image_size : int
    output_path : str
    checkpoint_path : str

    def __post_init__(self):
        if self.image_size <= 0:
            raise ValueError(f"image_size must be greater than one, got {self.image_size}")

@dataclass(frozen=True, slots=True)
class NoiseParameters:
    min_snr : float
    max_snr : float
    snr_variance : float
    enable_impulsive_noise : bool
    low_band_noise : bool

    def __post_init__(self):
        if self.min_snr > self.max_snr:
            raise ValueError("min_snr must be smaller than max_snr")

        if self.snr_variance < 0:
            raise ValueError("snr_variance must be positive")

@dataclass(frozen=True, slots=True)
class GenerationParameters:
    whistle_bank_path : str
    num_audio_samples: int
    output_dir : Path
    max_num_whistles : int

    def __post_init__(self):
        if self.num_audio_samples < 0:
            raise ValueError("num_audio_samples must be a positive integer")

        wav_path = self.output_dir / "wav"
        mask_path = self.output_dir / "mask"
        png_path = self.output_dir / "png"

        for directory in (self.output_dir, wav_path, mask_path, png_path):
            directory.mkdir(parents=True, exist_ok=True)

        if self.max_num_whistles < 0:
            raise ValueError(f"max_num_whistles must be positive, got {self.max_num_whistles}")

    @property
    def wav_count(self):
        wav_dir = self.output_dir / "wav"
        return sum(1 for p in wav_dir.iterdir()) if wav_dir.exists() else 0

@dataclass(frozen=True, slots=True)
class Parameters:
    audio : AudioParameters
    stft : STFTParamters
    array : ArrayParameters
    network : NetworkParameters
    noise : NoiseParameters
    generation : GenerationParameters

    def __post_init__(self):
        if self.audio.duration is not None:
            if not self.audio.duration <= self.max_duration:
                raise ValueError("duration is greater than maximum allowed duration")

    @classmethod
    def from_json(cls, path: str | Path):
        path = Path(path)

        with path.open() as file: 
            data = json.load(file)

        network_data = data["network_parameters"].copy()
        network_data["checkpoint_path"] = Path(network_data["checkpoint_path"])

        generation_data= data["generation_parameters"].copy()
        generation_data["whistle_bank_path"] = Path(generation_data["whistle_bank_path"]) 
        generation_data["output_dir"]= Path(generation_data["output_dir"]) 

        return cls(
            audio=AudioParameters(**data["audio_parameters"]),
            stft=STFTParamters(**data["stft_parameters"]),
            array=ArrayParameters(**data["array_parameters"]),
            network=NetworkParameters(**network_data),
            noise=NoiseParameters(**data["noise_parameters"]),
            generation=GenerationParameters(**generation_data)
        )

    @property
    def max_duration(self):
        #TODO: adapt max_duration in function of the stft apraemeters udsed
        return ((self.network.image_size - 1)*self.stft.hop_length + self.stft.n_fft)/self.audio.sampling_rate

    @property
    def stft_shape(self) -> tuple[int, int]:
        n = self.audio.num_samples
        n_freqs = self.stft.num_frequency_bins_between(
            self.audio.sampling_rate,
            self.audio.min_freq,
            self.audio.max_freq
        )

        n_times = self.stft.num_time_bins(n)
        return n_freqs, n_times

    