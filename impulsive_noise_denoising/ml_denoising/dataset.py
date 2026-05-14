import json
from pathlib import Path
import lightning as L
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Generator
from impulsive_noise_denoising.wav_reader import read_wav_file
import numpy as np

class ImpulsiveNoiseDataset(Dataset):
    def __init__(self, output_path):
        absolute_output_path = Path(output_path).resolve()
        if not absolute_output_path.is_dir():
            raise ValueError(f"the provided output_path {output_path} does not exist")
        
        wav_folder = absolute_output_path / "wavs"
        label_folder = absolute_output_path / "labels"

        if not wav_folder.is_dir():
            raise FileNotFoundError(f"the provided output_path {output_path} does not contain wavs subfolder")
        
        if not label_folder.is_dir():
            raise FileNotFoundError(f"the provided output_path {output_path} does not contain labels subfolder")
        
        self.wav_dir = wav_folder
        self.label_dir = label_folder
        self.samples = []

        for label_path in sorted(self.label_dir.glob("*.labels.json")):
            with label_path.open("r", encoding="utf-8") as file:
                label_data = json.load(file)

            segment_name = label_data.get("segment_name")
            if not segment_name:
                raise ValueError(f"{label_path} does not contain a segment_name")

            wav_path = self.wav_dir / f"{segment_name}.wav"
            if not wav_path.is_file():
                raise FileNotFoundError(
                    f"Could not find wav for segment_name={segment_name}. "
                    f"Expected {wav_path}"
                )

            self.samples.append(
                {
                    "wav_path": wav_path,
                    "label_path": label_path,
                    "segment_name": segment_name,
                    "labels": label_data.get("labels", []),
                    "sampling_rate": label_data.get("sampling_rate"),
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_signal, frame_rate = read_wav_file(sample["wav_path"], num_canals=1)
        impulsive_mask = np.zeros_like(input_signal, dtype=np.float32)

        for start_time, end_time in sample["labels"]:
            start_idx = int(round(start_time * frame_rate))
            end_idx = int(round(end_time * frame_rate))
            impulsive_mask[..., start_idx:end_idx] = 1.0

        return {
            "input_signal": input_signal,
            "impulsive_mask": impulsive_mask,
            "frame_rate": frame_rate,
            "wav_path": str(sample["wav_path"]),
            "label_path": str(sample["label_path"]),
            "segment_name": sample["segment_name"],
        }


class ImpulsiveNoiseDataModule(L.LightningDataModule):
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
            dataset = ImpulsiveNoiseDataset(self.dataset_path)
            split_lengths = self._split_lengths(len(dataset))
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                dataset,
                split_lengths,
                generator=Generator().manual_seed(self.seed),
            )

        elif self.explicit_split_mode:
            self.train_dataset = ImpulsiveNoiseDataset(self.train_path)
            self.val_dataset = ImpulsiveNoiseDataset(self.val_path)
            self.test_dataset = ImpulsiveNoiseDataset(self.test_path)

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
