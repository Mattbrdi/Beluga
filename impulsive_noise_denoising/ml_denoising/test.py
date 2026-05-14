import numpy as np
import sys 
from pathlib import Path 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch


from utils.plot import plot_mask_1D
        
test_data_path = r"C:\Users\amine\Desktop\Canada\Beluga\impulsive_noise_denoising\ml_denoising\data\dataset_1\output"
test_data = ImpulsiveNoiseDataset(test_data_path)

test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

data_dict = next(iter(test_dataloader))

input_signal, impulsive_mask, frame_rate, wav_path, label_path, segment_name = data_dict.values()



if __name__ == "__main__":
    output_path = Path(__file__).parent / "data" / "output"
    dataset = ImpulsiveNoiseDataset(output_path)
    print(f"dataset length: {len(dataset)}")
    item = dataset[0]
    print(item["segment_name"])
    print(item["input_signal"].shape, item["impulsive_mask"].shape)

    plot_mask_1D(input_signal.numpy()[0], frame_rate.numpy()[0], impulsive_mask.numpy()[0])