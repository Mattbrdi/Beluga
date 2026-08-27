from __future__ import annotations
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import numpy.random as rd
from itertools import combinations

from time_frequency_mask.config import Parameters, STFTParamters
from time_frequency_mask.data_generation.core.multi_canal import set_channels_tdoas
from time_frequency_mask.data_generation.core.power_computation import set_std_from_snr
from time_frequency_mask.data_generation.core.mask_visibility import (
    update_mask_for_noise,
)
from time_frequency_mask.data_generation.generators.noise_generator import (
    gaussian_noise_generator,
)
from time_frequency_mask.data_generation.io.data_parser import (
    read_wav_file,
    save_mask,
    save_wav_file,
    save_stft_png,
)
from time_frequency_mask.data_generation.models.mask import AudioMask, WhistleMask


class AudioSample:
    def __init__(
        self,
        waveform: NDArray[np.float64],
        mask: WhistleMask,
        sampling_rate: float,
        stft_parameters: STFTParamters,
    ):
        n_fft, hop_length = stft_parameters.n_fft, stft_parameters.hop_length
        n_times_spectro = int(1 + np.floor(len(waveform) - n_fft) / hop_length)
        if n_times_spectro != mask.data.shape[1]:
            raise ValueError(
                f"Non matching times bins got {n_times_spectro} for waveform and {mask.data.shape[1]} for label"
            )

        self.waveform = waveform
        self.mask = mask
        self.sampling_rate = sampling_rate
        self.stft_parameters = stft_parameters

    @classmethod
    def from_path(cls, waveform_path : str, mask_path : str, stft_parameters : STFTParamters):
        waveform, sampling_rate = read_wav_file(waveform_path, num_canals=1)
        mask = WhistleMask.from_path(mask_path, sampling_rate)
        return cls(waveform, mask, sampling_rate, stft_parameters)


class Whistle(AudioSample):
    def __init__(
        self,
        waveform: NDArray[np.float64],
        mask: WhistleMask,
        sampling_rate: float,
        stft_parameters: STFTParamters,
        start_time=None,
    ):
        super().__init__(waveform, mask, sampling_rate, stft_parameters)
        self.start_time = start_time

    @classmethod
    def from_path(cls, waveform_path, mask_path, stft_parameters, start_time=None):
        waveform, sampling_rate = read_wav_file(waveform_path, num_canals=1)
        mask = WhistleMask.from_path(mask_path, sampling_rate)
        return cls(waveform, mask, sampling_rate, stft_parameters, start_time)

    def place(self, num_samples: int, n_freqs: int, n_times: int) -> LabeledAudioSample:
        if self.start_time is None:
            raise ValueError("Error placing Whistle without providing start_time")

        if self.start_time >= num_samples / self.sampling_rate:
            raise ValueError(
                f"provided start time {self.start_time} is greater than DURATION {num_samples / self.sampling_rate}"
            )

        start_index = int(np.floor(self.sampling_rate * self.start_time))
        whistle_width = self.waveform.shape[0]
        dst_start = max(0, start_index)
        src_start = max(0, -start_index)

        available_dst = num_samples - dst_start
        available_src = whistle_width - src_start

        placed_width = min(available_dst, available_src)

        if placed_width <= 0:
            return LabeledAudioSample.from_empty_wav_and_mask_size(
                self.sampling_rate,
                num_samples / self.sampling_rate,
                self.stft_parameters,
                n_freqs,
                n_times,
            )

        dst_end = dst_start + placed_width
        src_end = src_start + placed_width

        waveform = np.zeros(num_samples, dtype=np.float64)
        waveform[dst_start:dst_end] = self.waveform[src_start:src_end]

        mask = self.mask.place(self.start_time, n_freqs, n_times, num_samples)
        return LabeledAudioSample(
            waveform, mask, self.sampling_rate, self.stft_parameters
        )


class LabeledAudioSample(AudioSample):
    def __init__(
        self,
        waveform: NDArray[np.float64],
        mask: AudioMask,
        sampling_rate: float,
        stft_parameters: STFTParamters,
    ):
        super().__init__(waveform, mask, sampling_rate, stft_parameters)

    @classmethod
    def from_empty_wav_and_mask_size(
        cls,
        sampling_rate : float,
        duration : float,
        stft_parameters : STFTParamters,
        n_freqs : int,
        n_times : int
    ):
        waveform = np.zeros(shape=int(duration * sampling_rate), dtype=np.float64)

        mask = AudioMask.create_empty_mask(n_freqs, n_times, sampling_rate)
        return cls(waveform, mask, sampling_rate, stft_parameters)

    @classmethod
    def from_empty_wav(
        cls, parameters: Parameters, duration: float
    ) -> LabeledAudioSample:
        sampling_rate = parameters.audio.sampling_rate
        n_times = parameters.stft.num_time_bins(int(sampling_rate * duration))
        n_freqs = parameters.stft.num_frequency_bins_between(
            sampling_rate, parameters.audio.min_freq, parameters.audio.max_freq
        )

        waveform = np.zeros(shape=int(duration * sampling_rate), dtype=np.float64)

        mask = AudioMask.create_empty_mask(n_freqs, n_times, sampling_rate)
        return cls(waveform, mask, sampling_rate, parameters.stft)

    def __add__(self, other: AudioSample) -> LabeledAudioSample:
        if isinstance(other, Whistle):
            if other.start_time is None:
                raise ValueError(
                    f"Error no start_time provided for whistle added to LabeledAudioSample"
                )

            placed_whistle = other.place(
                len(self.waveform), self.mask.data.shape[0], self.mask.data.shape[1]
            )
            waveform = self.waveform + placed_whistle.waveform
            mask = self.mask + placed_whistle.mask

        elif isinstance(other, LabeledAudioSample):
            if self.sampling_rate != other.sampling_rate:
                raise ValueError("Sampling rates do not match")
            if self.stft_parameters != other.stft_parameters:
                raise ValueError("STFT parameters do not match")
            if self.waveform.shape != other.waveform.shape:
                raise ValueError("Waveform shapes do not match")

            waveform = self.waveform + other.waveform
            mask = self.mask + other.mask

        else:
            return NotImplemented

        return LabeledAudioSample(
            waveform, mask, self.sampling_rate, self.stft_parameters
        )


class TetrahedraAudioSample:
    def __init__(
        self,
        labeled_audio_sample_list: list[LabeledAudioSample],
        sampling_rate: int,
        num_microphones: int,
    ):
        if not len(labeled_audio_sample_list) == num_microphones:
            raise ValueError(
                f"Labeled_audio_sample_list is of length {len(labeled_audio_sample_list)} instead of {num_microphones}"
            )

        for i in range(num_microphones):
            if (
                not labeled_audio_sample_list[0].sampling_rate
                == labeled_audio_sample_list[i].sampling_rate
            ):
                raise ValueError(f"Non matching sampling rates")
            if not len(labeled_audio_sample_list[0].waveform) == len(
                labeled_audio_sample_list[i].waveform
            ):
                raise ValueError(f"Non matching labeled_audio_sample lengths")

        if not labeled_audio_sample_list[0].sampling_rate == sampling_rate:
            raise ValueError(
                f"labeled_audio_samples have sampling rate {labeled_audio_sample_list[0].sampling_rate} instead of {sampling_rate}"
            )

        self.waveforms = np.array(
            [
                labeled_audio_sample.waveform
                for labeled_audio_sample in labeled_audio_sample_list
            ],
            dtype=np.float64,
        )
        self.masks = np.array(
            [
                labeled_audio_sample.mask
                for labeled_audio_sample in labeled_audio_sample_list
            ]
        )

        self.num_microphones = num_microphones

        self.sampling_rate = sampling_rate
        self.duration = len(self.waveforms[0]) / self.sampling_rate

        self.shifted_waveforms = self.waveforms.copy()
        self.shifted_masks = self.masks.copy()

    @classmethod
    def from_single_labeled_audio_sample(
        cls, labeled_audio_sample: LabeledAudioSample, num_microphones: int
    ) -> TetrahedraAudioSample:
        labeled_audio_sample_list = [
            labeled_audio_sample for _ in range(num_microphones)
        ]
        return cls(
            labeled_audio_sample_list,
            labeled_audio_sample.sampling_rate,
            num_microphones,
        )

    def set_tdoas(self, parameters: Parameters, shifts: list[float]):
        mask_data = [mask.data for mask in self.masks]

        shifted_waveforms, shifted_masks = set_channels_tdoas(
            self.waveforms,
            mask_data,
            parameters,
            shifts,
        )

        self.shifted_waveforms = np.array(shifted_waveforms, dtype=np.float64)
        self.shifted_masks = np.array(
            [
                AudioMask(mask, mask.shape[0], mask.shape[1], self.sampling_rate)
                for mask in shifted_masks
            ]
        )

    def set_common_impulsive_noise(self):
        pass

    def mask_unifier(self):
        combined_mask_data = self.shifted_masks[0].data

        stacked_masks = np.stack([mask.data for mask in self.shifted_masks], axis=0)
        combined_mask_data = np.count_nonzero(stacked_masks, axis=0) >= max(
            1, self.num_microphones - 1
        )

        for i in range(len(self.shifted_masks)):
            self.shifted_masks[i].data = combined_mask_data

    def set_gaussian_noise(self, snrs_db: list[float], parameters: Parameters):
        for i in range(self.num_microphones):
            shifted_waveform, mask = self.shifted_waveforms[i], self.shifted_masks[i]

            noise_std = set_std_from_snr(
                shifted_waveform, mask.data, parameters, snrs_db[i]
            )

            noise = gaussian_noise_generator(
                std=noise_std,
                duration=self.duration,
                sampling_rate=self.sampling_rate,
                is_low_band_noise=parameters.noise.low_band_noise,
            )

            new_mask_data = update_mask_for_noise(
                shifted_waveform, noise, mask.data, parameters
            )

            self.shifted_waveforms[i] += noise
            self.shifted_masks[i].data = new_mask_data

    def save(self, output_path: str, stem: str, parameters: Parameters):
        path = Path(output_path)

        png_output_path = path / "png" / f"{stem}.png"
        mask_output_path = path / "mask" / f"{stem}.png"
        audio_array_output_path = path / "wav" / f"{stem}.wav"
        # Save png:
        save_stft_png(self.shifted_waveforms, str(png_output_path), parameters)
        self.mask_unifier()
        save_mask(self.shifted_masks[0].data, str(mask_output_path))

        # Save Waveform:
        save_wav_file(self.shifted_waveforms, str(audio_array_output_path), self.sampling_rate, parameters.array.num_mics)
