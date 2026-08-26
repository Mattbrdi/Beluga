import numpy as np
import argparse
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from time_frequency_mask.data_generation.io.data_parser import read_wav_file
from time_frequency_mask.stft import frequency_band, scipy_spectrogram
from time_frequency_mask.config import Parameters
from matplotlib.cm import magma
from PIL import Image
PATH = r"C:\Users\amine\Desktop\Canada\Beluga\time_frequency_mask\data_generation\data\input\whistle2"

def parse_args():
    parser = argparse.ArgumentParser(description="STFT plot for polygon labeling of Whistles")
    parser.add_argument(
        "--config", help="Provide configuration file", type=str, required=True,
    )
    parser.add_argument(
        "--wav-dir", help="Provide the whistle wav directory", type=str, required=True,
    )
    parser.add_argument(
        "--output-dir", help="Provide the whistle wav directory", type=str, required=True,
    )
    return parser.parse_args()

def main(parameters : Parameters, wav_dir : Path, output_dir : Path):
    for waveform_path in Path(wav_dir).glob("*.wav"):
        waveform_name = str(waveform_path).split('.')[0]
        waveform, sampling_rate = read_wav_file(str(waveform_path), num_canals=1)

        freqs, times, D = scipy_spectrogram(waveform, sampling_rate, parameters.stft)
        freqs, D = frequency_band(freqs, D, parameters.audio.min_freq, parameters.audio.max_freq)
        D = D - np.min(D)
        
        D = D / np.percentile(np.abs(D),99)
    
        D = np.clip(D, 0,1)
    
        D = np.array(magma(D))
        
        D = np.round(D*255)

        D = D.astype(np.uint8)
        
        # 2. Convert the NumPy array to a Pillow Image object
        img = Image.fromarray(D)
        output_folder = Path(output_dir) / "png"
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / f"{waveform_path.stem}.png"    
        img.save(output_path)

if __name__ == "__main__":
    args = parse_args()
    parameters = Parameters.from_json(args.config)
    wav_dir = Path(args.wav_dir)
    output_dir = Path(args.output_dir)
    main(parameters, wav_dir, output_dir)
