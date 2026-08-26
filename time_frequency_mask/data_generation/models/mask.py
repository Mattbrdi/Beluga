from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from time_frequency_mask.config import Parameters
from time_frequency_mask.data_generation.io.data_parser import read_image_file


class Mask:
    def __init__(self, data: NDArray[np.uint8], sampling_rate: float):
        self.sampling_rate = sampling_rate
        self.data = data
        self.height, self.width = data.shape

    @classmethod
    def from_path(cls, file_path: str, sampling_rate: float) -> Mask:
        return cls(read_image_file(file_path), sampling_rate)

    def apply_mask(self, audio_array: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply a Mask to an audio array"""
        pass

    def __add__(self, other: Mask) -> Mask:
        """Or the two Masks by overloading the + operator"""
        if isinstance(other, Mask):
            pass
        else:
            return NotImplemented

    def inverse_mask(self) -> Mask:
        """Inverse 0 and 1 in the mask"""
        pass

    def save_mask_to_txt(self):
        pass


class WhistleMask(Mask):
    def __init__(self, data, sampling_rate):
        super().__init__(data, sampling_rate)

    def place(
        self, start_time: float, n_freqs: int, n_times: int, num_samples: int
    ) -> AudioMask:
        duration = num_samples / self.sampling_rate

        if start_time >= duration:
            raise ValueError(
                f"provided start time {start_time} is greater than duration {duration}"
            )

        start_index = int(np.floor(n_times * start_time / duration))
        whistle_width = self.width
        dst_start = max(0, start_index)
        src_start = max(0, -start_index)

        available_dst = n_times - dst_start
        available_src = whistle_width - src_start

        placed_width = min(available_dst, available_src)

        if placed_width <= 0:
            return AudioMask.create_empty_mask(n_freqs, n_times, self.sampling_rate)

        dst_end = dst_start + placed_width
        src_end = src_start + placed_width

        data = np.zeros((n_freqs, n_times), dtype=np.uint8)
        data[:, dst_start:dst_end] = self.data[:, src_start:src_end]
        return AudioMask(
            data=data,
            n_freqs=n_freqs,
            n_times=n_times,
            sampling_rate=self.sampling_rate,
        )


class AudioMask(Mask):
    def __init__(self, data, n_freqs, n_times, sampling_rate):
        super().__init__(data, sampling_rate)

        if np.shape(self.data) != (n_freqs, n_times):
            raise ValueError(
                f"incorrect data shape got {self.data.shape} instead of {(n_freqs, n_times)}"
            )

    @classmethod
    def create_empty_mask(
        cls, n_freqs: int, n_times: int, sampling_rate: float
    ) -> AudioMask:
        data = np.zeros((n_freqs, n_times), dtype=np.uint8)
        return cls(data, n_freqs, n_times, sampling_rate)

    def __add__(self, other: Mask) -> AudioMask:
        if isinstance(other, AudioMask):
            if self.sampling_rate != other.sampling_rate:
                raise ValueError(
                    f"sampling_rate do not match first is {self.sampling_rate} and second is {other.sampling_rate}"
                )

            if self.data.shape != other.data.shape:
                raise ValueError(
                    f"Incorrect array format self is {self.data.shape} and other is {other.data.shape}"
                )
            return AudioMask(
                self.data | other.data,
                self.data.shape[0],
                self.data.shape[1],
                self.sampling_rate,
            )
        elif isinstance(other, WhistleMask):
            raise TypeError(
                f"Cannot add {type(self).__name__} and {type(other).__name__}: "
                "mismatched specializations."
            )
        else:
            return NotImplemented
