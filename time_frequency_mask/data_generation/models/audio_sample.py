from __future__ import annotations
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from time_frequency_mask.configuration import SAMPLING_RATE, DURATION, N_FFT, HOP_LENGTH, MAX_TDOA, MIN_FREQ, MAX_FREQ
from time_frequency_mask.stft import compute_power_from_waveform_and_mask, scipy_spectrogram, frequency_band
from time_frequency_mask.data_generation.core.multi_canal import set_channels_tdoas
from time_frequency_mask.data_generation.generators.noise_generator import gaussian_noise_generator
from time_frequency_mask.data_generation.io.data_parser import read_wav_file, save_mask, save_wav_file, save_stft_png
from time_frequency_mask.data_generation.models.mask import AudioMask, WhistleMask

class AudioSample:
    def __init__(self, waveform : NDArray[np.float64], mask : WhistleMask, sampling_rate : float):
        if sampling_rate != SAMPLING_RATE:
            raise ValueError(f"incorrect sampling_rate got {sampling_rate} instead of {SAMPLING_RATE}")
        N_TIMES_SPECTRO = int(1 + np.floor(len(waveform) - N_FFT) / HOP_LENGTH)
        if N_TIMES_SPECTRO != mask.data.shape[1]:
            raise ValueError(f"Non matching times bins got {N_TIMES_SPECTRO} for waveform and {mask.data.shape[1]} for label")

        self.waveform = waveform
        self.mask = mask
        self.sampling_rate = sampling_rate
        self.duration = len(waveform) / sampling_rate

    @classmethod
    def from_path(cls, waveform_path, mask_path):
        waveform, sampling_rate = read_wav_file(waveform_path, num_canals=1)
        mask = WhistleMask.from_path(mask_path)
        return cls(waveform, mask, sampling_rate)
    
class Whistle(AudioSample):
    def __init__(self, waveform : NDArray[np.float64], mask : WhistleMask, sampling_rate : float, start_time = None):
        super().__init__(waveform, mask, sampling_rate)
        self.start_time = start_time

    @classmethod
    def from_path(cls, waveform_path, mask_path, start_time = None):
        waveform, sampling_rate = read_wav_file(waveform_path, num_canals=1)
        mask = WhistleMask.from_path(mask_path)
        return cls(waveform, mask, sampling_rate, start_time)

    def place(self) -> LabeledAudioSample:
        if self.start_time is None:
            raise ValueError("Error placing Whistle without providing start_time")
        
        if self.start_time >= DURATION:
            raise ValueError(f"provided start time {self.start_time} is greater than DURATION {DURATION}")
        
        N = int(DURATION * SAMPLING_RATE)

        start_index = int(np.floor(SAMPLING_RATE * self.start_time))
        whistle_width = self.waveform.shape[0]
        dst_start = max(0, start_index)
        src_start = max(0, -start_index)

        available_dst = N - dst_start
        available_src = whistle_width - src_start
        
        placed_width = min(available_dst, available_src)

        if placed_width <= 0:
            return LabeledAudioSample.from_empty_wav()
        
        dst_end = dst_start + placed_width
        src_end = src_start + placed_width

        waveform = np.zeros(N, dtype=np.float64)
        waveform[ dst_start: dst_end] = self.waveform[src_start:src_end]

        mask = self.mask.place(self.start_time)
        return LabeledAudioSample(waveform, mask, self.sampling_rate)

class LabeledAudioSample(AudioSample):
    def __init__(self, waveform: NDArray[np.float64], mask : AudioMask, sampling_rate : float):
        super().__init__(waveform, mask, sampling_rate)

        if len(self.waveform) != DURATION * SAMPLING_RATE:
            raise ValueError(f"wrong waveform duration got {len(self.waveform)} samples instead of {DURATION * SAMPLING_RATE}")

    @classmethod
    def from_empty_wav(cls) -> LabeledAudioSample:
        waveform = np.zeros(shape=(DURATION * SAMPLING_RATE), dtype=np.float64)
        mask = AudioMask.create_empty_mask(SAMPLING_RATE)
        return cls(waveform, mask, SAMPLING_RATE)

    def __add__(self, other : AudioSample) -> LabeledAudioSample:
        if isinstance(other, Whistle):
            if other.start_time is None:
                raise ValueError(f"Error no start_time provided for whistle added to LabeledAudioSample")
            
            placed_whistle = other.place()
            waveform = self.waveform + placed_whistle.waveform
            mask = self.mask + placed_whistle.mask
        
        elif isinstance(other, LabeledAudioSample):
            waveform = self.waveform + other.waveform
            mask = self.mask + placed_whistle.mask

        else:
            raise NotImplemented
        
        return LabeledAudioSample(waveform, mask, self.sampling_rate)

class TetrahedraAudioSample:
    def __init__(self, labeled_audio_sample_list : list[LabeledAudioSample]):
        if not len(labeled_audio_sample_list) == 4:
            raise ValueError(f"labeled_audio_sample_list is of length {len(labeled_audio_sample_list)} instead of 4")
        
        if not labeled_audio_sample_list[0].sampling_rate == labeled_audio_sample_list[1].sampling_rate == labeled_audio_sample_list[2].sampling_rate == labeled_audio_sample_list[3].sampling_rate: 
            raise ValueError(f"got not matching sampling rates")

        if not labeled_audio_sample_list[0].sampling_rate == SAMPLING_RATE:
            raise ValueError(f"labeled_audio_samples have sampling rate {labeled_audio_sample_list[0].sampling_rate} instead of {SAMPLING_RATE}")
        
        self.waveforms = np.array([labeled_audio_sample.waveform for labeled_audio_sample in labeled_audio_sample_list], dtype=np.float64)
        self.masks = np.array([labeled_audio_sample.mask for labeled_audio_sample in labeled_audio_sample_list])

        self.sampling_rate = labeled_audio_sample_list[0].sampling_rate

        self.shifted_waveforms = self.waveforms.copy()
        self.shifted_masks = self.masks.copy()

        self.stft = self.set_stft()

    @classmethod
    def from_single_labeled_audio_sample(cls, labeled_audio_sample : LabeledAudioSample) -> TetrahedraAudioSample:
        labeled_audio_sample_list = [labeled_audio_sample for _ in range(4)]
        return cls(labeled_audio_sample_list)

    def set_stft(self):
        freqs, times, D = scipy_spectrogram(self.waveforms[0], SAMPLING_RATE)
        
        freqs, D = frequency_band(freqs, D, MIN_FREQ, MAX_FREQ)
        
        D = D - np.min(D)
        D = D / np.percentile(np.abs(D),99)
        
        return D

    def set_tdoas(self, shifts: list[float]):
        mask_data = [mask.data for mask in self.masks]

        shifted_waveforms, shifted_masks = set_channels_tdoas(
            self.waveforms,
            mask_data,
            shifts,
        )

        self.shifted_waveforms = np.array(shifted_waveforms, dtype=np.float64)
        self.shifted_masks = np.array([
            AudioMask(mask, self.sampling_rate) for mask in shifted_masks
        ])

    def set_common_impulsive_noise(self):
        pass

    def set_gaussian_noise(self, snrs_db : list[float], change_mask : bool = True):
        for i in range(4):
            shifted_waveforms, mask = self.shifted_waveforms[i], self.masks[i].data

            signal_power = compute_power_from_waveform_and_mask(shifted_waveforms, mask)

            snr = 10 ** (snrs_db[i] / 10)
            noise_std = np.sqrt(signal_power / snr)

            noise = gaussian_noise_generator(noise_std)

            # freqs, times, noise_stft = scipy_spectrogram(self.waveforms[0], SAMPLING_RATE)
        
            # freqs, noise_stft = frequency_band(freqs, noise_stft, MIN_FREQ, MAX_FREQ)
            # if change_mask:
            #     mask_data = self.shifted_masks[0].data
            #     waveform = self.shifted_waveforms[i]

            #     condition = (
            #         (mask_data >= 1)
            #         & (np.abs(noise_stft) > 2 * np.abs(self.stft))
            #     )
            #     mask_data[condition] = 0

            self.shifted_waveforms[i] += noise

    def save(self, output_path : str, stem : str):
        path = Path(output_path)

        png_output_path = path / "png" / f"{stem}.png"
        png_output_path.parent.mkdir(parents=True, exist_ok=True)
        mask_output_path = path / "mask" / f"{stem}.png"
        mask_output_path.parent.mkdir(parents=True, exist_ok=True)
        audio_array_output_path = path / "wav" / f"{stem}.wav"
        audio_array_output_path.parent.mkdir(parents=True, exist_ok=True)
        # Save png:
        save_stft_png(self.shifted_waveforms, str(png_output_path))

        # Save mask:
        save_mask(self.shifted_masks[0].data, str(mask_output_path))

        # Save Waveform:
        save_wav_file(self.shifted_waveforms, str(audio_array_output_path))







            
