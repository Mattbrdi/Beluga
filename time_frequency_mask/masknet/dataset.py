from pathlib import Path
from PIL import Image

import numpy as np

import lightning as L
from torch import Generator
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.signal import istft

from time_frequency_mask.configuration import (
    HOP_LENGTH,
    IMAGE_SIZE,
    N_FFT,
    N_FREQS,
    SAMPLING_RATE,
)
from time_frequency_mask.stft import frequency_band, scipy_stft_complex
from time_frequency_mask.data_generation.core.preprocess import bandpass_filter
from time_frequency_mask.data_generation.io.data_parser import read_wav_file


def translate_waveform(waveform, shift):
    freqs, times, D_full = scipy_stft_complex(waveform)

    F, T = D_full.shape

    src_start = max(0, -shift)
    dst_start = max(0, shift)

    available_dst = F - dst_start
    available_src = F - src_start

    placed_width = min(available_dst, available_src)

    if placed_width <= 0:
        print("debug")
        return np.zeros_like(waveform)

    src_end = src_start + placed_width
    dst_end = dst_start + placed_width

    D_full_translated = np.zeros_like(D_full)

    D_full_translated[dst_start:dst_end] = D_full[src_start:src_end]

    _, waveform_translated = istft(D_full_translated, SAMPLING_RATE, 'hann', noverlap=N_FFT - HOP_LENGTH, nfft=N_FFT)
    waveform_translated = bandpass_filter(waveform_translated, SAMPLING_RATE)

    return np.pad(waveform_translated, (0, len(waveform)-len(waveform_translated)))

def translate_mask(mask, shift):
    src_start = max(0, -shift)
    dst_start = max(0, shift)

    available_dst = N_FREQS - dst_start
    available_src = N_FREQS - src_start

    placed_width = min(available_dst, available_src)

    if placed_width <= 0:
        return np.zeros_like(mask)

    src_end = src_start + placed_width
    dst_end = dst_start + placed_width

    new_mask = np.zeros_like(mask)
    new_mask[dst_start:dst_end, :] = mask[src_start:src_end, :]
    return new_mask

MICROPHONE_PAIRS = tuple(
    (first, second)
    for first in range(4)
    for second in range(first + 1, 4)
)


def _normalize_magnitude(magnitude: np.ndarray) -> np.ndarray:
    """Match the legacy per-microphone percentile normalization."""
    normalized = np.asarray(magnitude, dtype=np.float64).copy()
    if normalized.size == 0 or not np.all(np.isfinite(normalized)):
        raise ValueError("The STFT magnitude must be finite and non-empty.")
    if np.max(np.abs(normalized)) <= 0:
        return np.zeros_like(normalized, dtype=np.float32)
    normalized -= np.min(normalized)
    scale = float(np.percentile(normalized, 99))
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros_like(normalized, dtype=np.float32)
    return np.clip(normalized / scale, 0.0, 1.0).astype(np.float32)


def build_spectrogram_features(
    audio_array: np.ndarray,
    *,
    phase_aware: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build four magnitude or sixteen magnitude-plus-IPD channels."""
    audio = np.asarray(audio_array)
    if audio.ndim != 2 or audio.shape[0] != 4:
        raise ValueError(
            f"Expected four-channel audio shape (4, samples), got {audio.shape}."
        )

    complex_stfts: list[np.ndarray] = []
    frequencies: np.ndarray | None = None
    times: np.ndarray | None = None
    for microphone in audio:
        channel_frequencies, channel_times, channel_stft = scipy_stft_complex(
            microphone
        )
        channel_frequencies, channel_stft = frequency_band(
            channel_frequencies, channel_stft
        )
        if frequencies is None:
            frequencies = channel_frequencies
            times = channel_times
        elif not np.array_equal(channel_frequencies, frequencies) or not np.array_equal(
            channel_times, times
        ):
            raise ValueError("Microphone STFT grids do not match.")
        complex_stfts.append(channel_stft)

    assert frequencies is not None and times is not None
    complex_stft = np.stack(complex_stfts, axis=0)
    magnitude_features = np.stack(
        [_normalize_magnitude(2.0 * np.abs(value)) for value in complex_stft],
        axis=0,
    )
    if not phase_aware:
        return magnitude_features, frequencies, times

    phase_features: list[np.ndarray] = []
    epsilon = np.finfo(complex_stft.real.dtype).eps
    for first, second in MICROPHONE_PAIRS:
        cross_spectrum = complex_stft[first] * complex_stft[second].conj()
        magnitude = np.abs(cross_spectrum)
        normalized_cross_spectrum = np.divide(
            cross_spectrum,
            magnitude,
            out=np.zeros_like(cross_spectrum),
            where=magnitude > epsilon,
        )
        phase_features.extend(
            (
                normalized_cross_spectrum.real.astype(np.float32),
                normalized_cross_spectrum.imag.astype(np.float32),
            )
        )
    features = np.concatenate(
        (magnitude_features, np.stack(phase_features, axis=0)), axis=0
    )
    return features.astype(np.float32, copy=False), frequencies, times


def _center_pad(array: np.ndarray, image_size: int = IMAGE_SIZE) -> np.ndarray:
    if array.ndim < 2:
        raise ValueError("Expected frequency and time dimensions.")
    frequencies, times = array.shape[-2:]
    if frequencies > image_size or times > image_size:
        raise ValueError(
            f"Array shape {array.shape} exceeds image size "
            f"{(image_size, image_size)}."
        )
    frequency_padding = image_size - frequencies
    time_padding = image_size - times
    padding = [(0, 0)] * array.ndim
    padding[-2] = (
        frequency_padding // 2,
        frequency_padding - frequency_padding // 2,
    )
    padding[-1] = (time_padding // 2, time_padding - time_padding // 2)
    return np.pad(array, padding)


class WhistleMaskDataset(Dataset):
    """Load four-channel magnitude or sixteen-channel phase-aware features."""

    def __init__(self, output_path, phase_aware: bool = False):
        absolute_output_path = Path(output_path).resolve()
        if not absolute_output_path.is_dir():
            raise ValueError(f"the provided output_path {output_path} does not exist")

        wav_folder = absolute_output_path / "wav"
        mask_folder = absolute_output_path / "mask"
        if not wav_folder.is_dir():
            raise FileNotFoundError(
                f"the provided output_path {output_path} does not contain wavs subfolder"
            )
        if not mask_folder.is_dir():
            raise FileNotFoundError(
                f"the provided output_path {output_path} does not contain labels subfolder"
            )

        self.wav_dir = wav_folder
        self.mask_dir = mask_folder
        self.phase_aware = bool(phase_aware)
        self.n_input_channels = 16 if self.phase_aware else 4
        self.samples = []

        for mask_path in sorted(self.mask_dir.glob("*.png")):
            segment_name = mask_path.stem
            wav_path = self.wav_dir / f"{segment_name}.wav"
            if not wav_path.is_file():
                raise FileNotFoundError(
                    f"Could not find wav for segment_name={segment_name}. "
                    f"Expected {wav_path}"
                )
            self.samples.append(
                {
                    "wav_path": wav_path,
                    "mask_path": mask_path,
                    "segment_name": segment_name,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        with Image.open(sample["mask_path"]) as image:
            mask = np.asarray(image, dtype=np.uint8) > 0
        if mask.ndim != 2:
            raise ValueError(
                f"Expected a two-dimensional mask in {sample['mask_path']}, "
                f"got {mask.shape}."
            )
        input_signal, sampling_rate = read_wav_file(sample["wav_path"])
        if sampling_rate != SAMPLING_RATE:
            raise ValueError(
                f"Incorrect sampling rate got {sampling_rate}, "
                f"instead of {SAMPLING_RATE}"
            )
        features, _, _ = build_spectrogram_features(
            input_signal, phase_aware=self.phase_aware
        )
        if features.shape[-2:] != mask.shape:
            raise ValueError(
                f"Spectrogram shape {features.shape[-2:]} and mask shape "
                f"{mask.shape} are not compatible for {sample['segment_name']}."
            )
        features = _center_pad(features).astype(np.float32, copy=False)
        padded_mask = _center_pad(mask).astype(bool, copy=False)[np.newaxis, :]
        # return {
        #     "input_signal": input_signal,
        #     "stft": features,
        #     "mask": padded_mask,
        #     "wav_path": str(sample["wav_path"]),
        #     "mask_path": str(sample["mask_path"]),
        #     "segment_name": sample["segment_name"],
        #     "phase_aware": self.phase_aware,
        # }
        return {
                    "stft": features,
                    "mask": padded_mask,
                    "wav_path": str(sample["wav_path"]),
                    "mask_path": str(sample["mask_path"]),
                    "segment_name": sample["segment_name"],
                    "phase_aware": self.phase_aware,
                }
        


class WhistleMaskDataModule(L.LightningDataModule):
    def __init__(
            self,
            dataset_path=None,
            train_path=None,
            val_path=None,
            test_path=None,
            batch_size=4,
            train_val_test_split=(0.7, 0.15, 0.15),
            seed=42,
            num_workers=0,
            phase_aware=False,
        ):
        super().__init__()
        self.dataset_path = dataset_path
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.seed = seed
        self.num_workers = num_workers
        self.phase_aware = bool(phase_aware)

        self.single_dataset_mode = dataset_path is not None
        self.explicit_split_mode = any(path is not None for path in [train_path, val_path, test_path])

        if self.single_dataset_mode and self.explicit_split_mode:
            raise ValueError("Use either dataset_path or train/val/test paths, not both.")

        if not self.single_dataset_mode and not self.explicit_split_mode:
            raise ValueError("Provide either dataset_path or train/val/test paths.")

        if self.explicit_split_mode and not all([train_path, val_path, test_path]):
            raise ValueError("When using explicit splits, provide train_path, val_path, and test_path.")

    def setup(self, stage=None):
        if self.single_dataset_mode:
            dataset = WhistleMaskDataset(
                self.dataset_path, phase_aware=self.phase_aware
            )
            split_lengths = self._split_lengths(len(dataset))
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                dataset,
                split_lengths,
                generator=Generator().manual_seed(self.seed),
            )

        elif self.explicit_split_mode:
            self.train_dataset = WhistleMaskDataset(
                self.train_path, phase_aware=self.phase_aware
            )
            self.val_dataset = WhistleMaskDataset(
                self.val_path, phase_aware=self.phase_aware
            )
            self.test_dataset = WhistleMaskDataset(
                self.test_path, phase_aware=self.phase_aware
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def _split_lengths(self, dataset_size):
        train_fraction, val_fraction, test_fraction = self.train_val_test_split
        if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
            raise ValueError("train_val_test_split must sum to 1.0")

        train_length = int(dataset_size * train_fraction)
        val_length = int(dataset_size * val_fraction)
        test_length = dataset_size - train_length - val_length
        return [train_length, val_length, test_length]
