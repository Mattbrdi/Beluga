from pathlib import Path
from PIL import Image

import numpy as np
import numpy.random as rd 

import lightning as L
from torch import Generator
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.signal import stft, istft

from time_frequency_mask.configuration import N_FFT, HOP_LENGTH, SAMPLING_RATE, N_FFT, HOP_LENGTH, N_TIMES, N_FREQS, IMAGE_SIZE, IMAGE_SIZE
from time_frequency_mask.stft import frequency_band, scipy_stft_complex, scipy_spectrogram
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

class WhistleMaskDataset(Dataset):
    def __init__(self, output_path):
        absolute_output_path = Path(output_path).resolve()
        if not absolute_output_path.is_dir():
            raise ValueError(f"the provided output_path {output_path} does not exist")
        
        wav_folder = absolute_output_path / "wav"
        mask_folder = absolute_output_path / "mask"

        if not wav_folder.is_dir():
            raise FileNotFoundError(f"the provided output_path {output_path} does not contain wavs subfolder")
        
        if not mask_folder.is_dir():
            raise FileNotFoundError(f"the provided output_path {output_path} does not contain labels subfolder")
        
        self.wav_dir = wav_folder
        self.mask_dir = mask_folder
        self.samples = []

        for mask_path in sorted(self.mask_dir.glob("*.png")):
            data = Image.open(mask_path)
            width, height  = data.size
            data = np.array(data.get_flattened_data(), dtype=np.uint8)
            data = data > 0
            data = data.reshape((height, width))
            
            if height > IMAGE_SIZE or width >= IMAGE_SIZE:
                raise ValueError(f"incorrect shape got {data.shape} greater than {(IMAGE_SIZE, IMAGE_SIZE)}")

            diffX = IMAGE_SIZE - width 
            diffY = IMAGE_SIZE - height

            if diffX < 0 or diffY < 0:
                raise ValueError(f"D shape {data.shape} us larger than IMAGE_SIZE {IMAGE_SIZE}")

            data = np.pad(data, ((diffY //2, diffY - diffY //2), (diffX //2, diffX - diffX // 2)))
            data = data[np.newaxis, :]
            segment_name = mask_path.stem
            # segment_name = label_data.get("segment_name")
            # if not segment_name:
            #     raise ValueError(f"{label_path} does not contain a segment_name")

            wav_path = self.wav_dir / f"{segment_name}.wav"
            if not wav_path.is_file():
                raise FileNotFoundError(
                    f"Could not find wav for segment_name={segment_name}. "
                    f"Expected {wav_path}"
                )
            
            audio_array, sampling_rate = read_wav_file(wav_path)

            if sampling_rate != SAMPLING_RATE:
                raise ValueError(f"Incorrect sampling rate got {sampling_rate}, instead of {SAMPLING_RATE}")
            
            Ds = []

            freqs_list = []

            times_list = []

            for canal in audio_array:
                freqs, times, D = scipy_spectrogram(canal)
                # D = 20 * np.log10(np.maximum(2 * S, 1e-12))

                freqs, D = frequency_band(freqs, D)

                if np.max(np.abs(D)) > 0:
                    D = D - np.min(D)
                    D /= np.percentile(D,99)
                    D = np.clip(D, 0, 1)

                if D.shape != data.shape:
                    raise ValueError(
                        "Canal and mask data are not compatible."
                        f"Canal shape is {D.shape}."
                        f"mask shape is {data.shape}."
                    )

                n_freqs, n_times = D.shape
                if n_freqs > IMAGE_SIZE or n_times > IMAGE_SIZE:
                    raise ValueError(f"incorrect shape got {D.shape} greater than {(IMAGE_SIZE, IMAGE_SIZE)}")
                
                diffX = IMAGE_SIZE - n_times 
                diffY = IMAGE_SIZE - n_freqs

                if diffX < 0 or diffY < 0:
                    raise ValueError(f"D shape {D.shape} us larger than IMAGE_SIZE {IMAGE_SIZE}")

                D = np.pad(D, ((diffY // 2, diffY - diffY // 2), (diffX // 2, diffX - diffX // 2)))

                Ds.append(D)
                freqs_list.append(freqs)
                times_list.append(times)

            Ds = np.stack(Ds, axis=0)

        #TODO: add augmentation for this
        #     num_augment = rd.randint(0,3)

        #     for i in range(num_augment):
        #         shift = rd.randint(-N_FREQS //4, N_FREQS//4)
        #         mask_shifted = translate_mask(data, shift)
        #         Ds_shifted = []
        #         for D in Ds:
        #             D_shifted = translate_mask(D, shift)
        #             Ds_shifted.append(D_shifted)

        #         Ds_shifted = np.stack(Ds_shifted, axis=0)

        #     self.samples.append(
        #     {
        #         "wav_path": wav_path,
        #         "mask_path": mask_path,
        #         "segment_name": segment_name,
        #         "mask": mask_shifted,
        #         "stft": Ds_shifted 
        #     }
        # )


            self.samples.append(
                {
                    "wav_path": wav_path,
                    "mask_path": mask_path,
                    "segment_name": segment_name,
                    "mask": data,
                    "stft": Ds 
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_signal, frame_rate = read_wav_file(sample["wav_path"])
        mask = sample["mask"]

        return {
            "input_signal": input_signal,
            "stft": sample["stft"],
            "mask": mask,
            "wav_path": str(sample["wav_path"]),
            "mask_path": str(sample["mask_path"]),
            "segment_name": sample["segment_name"],
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
            num_workers=0
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
            dataset = WhistleMaskDataset(self.dataset_path)
            split_lengths = self._split_lengths(len(dataset))
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                dataset,
                split_lengths,
                generator=Generator().manual_seed(self.seed),
            )

        elif self.explicit_split_mode:
            self.train_dataset = WhistleMaskDataset(self.train_path)
            self.val_dataset = WhistleMaskDataset(self.val_path)
            self.test_dataset = WhistleMaskDataset(self.test_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def _split_lengths(self, dataset_size):
        train_fraction, val_fraction, test_fraction = self.train_val_test_split
        if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
            raise ValueError("train_val_test_split must sum to 1.0")

        train_length = int(dataset_size * train_fraction)
        val_length = int(dataset_size * val_fraction)
        test_length = dataset_size - train_length - val_length
        return [train_length, val_length, test_length]
