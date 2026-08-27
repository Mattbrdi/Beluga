import numpy as np
from numpy.typing import NDArray

import torch
import argparse


# import sys
# from pathlib import Path

# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))


from time_frequency_mask.config import Parameters, AudioParameters
from time_frequency_mask.plotter import plot_spectrogram_4D, plot_mask, plot_waveform_4D


from time_frequency_mask.masknet.models.spectro_mask_net_lightning import (
    SpectroMaskLightningModule,
)
from time_frequency_mask.masknet.models.spectro_mask_net import SpectroMaskNet

from time_frequency_mask.data_generation.core.preprocess import bandpass_filter
from time_frequency_mask.data_generation.io.data_parser import read_wav_file
from time_frequency_mask.masknet.dataset import build_spectrogram_features, _center_pad

# CKPT_PATH = r"C:\Users\amine\Desktop\Canada\Beluga\runs\spectro_mask_net\version_12\checkpoints\epoch=111-step=3584.ckpt"
# CKPT_PATH = r"C:\Users\amine\Desktop\Canada\Beluga\runs\spectro_mask_net\version_16\epoch=155-step=17316.ckpt"
# CKPT_PATH = r"C:\Users\amine\Desktop\Canada\Beluga\best-epoch=157-val_loss=0.2527.ckpt"
# CKPT_PATH = r"C:\Users\amine\Desktop\Canada\Beluga\last.ckpt"
WAV_PATH = r"C:\Users\amine\Desktop\Canada\Beluga\time_frequency_mask\data_generation\data\input\beluga_2026_test_duration_1_1s.wav"  # r"C:\Users\amine\Downloads\amine.wav"#
# WAV_PATH = r"C:\Users\amine\Desktop\Canada\Beluga\time_frequency_mask\data_generation\data\input\beluga_synth_02.wav"#r"C:\Users\amine\Downloads\amine.wav"#
WAV_PATH = r"C:\Users\amine\Desktop\Canada\Beluga\pipeline\test_data2026_all\data\beluga_2026_test_duration_5_4s.wav"
is_db = False


def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic beluga mask generator")
    parser.add_argument(
        "--config", help="Provide configuration file", type=str, required=True
    )
    parser.add_argument(
        "--wav-path", help="provide wav path", default=WAV_PATH, type=str
    )
    parser.add_argument(
        "--phase-aware",
        action="store_true",
        help=(
            "Use M*M input channels: four magnitudes and real/imaginary IPD "
            "features for all six microphone pairs."
        ),
    )
    return parser.parse_args()


def preprocess_waveform(
    waveform: NDArray[np.float64], audio_parameters: AudioParameters
) -> NDArray[np.float64]:
    waveform = bandpass_filter(waveform, audio_parameters.sampling_rate, audio_parameters.min_freq, audio_parameters.max_freq)
    scale = np.percentile(np.abs(waveform), 95)
    if scale == 0:
        scale = 1
    waveform = waveform / scale
    return waveform


def preprocess_torch(Ds: NDArray[np.float64], device):
    Ds = torch.from_numpy(Ds).float()
    x = Ds.unsqueeze(0).to(device)  # shape: [1, C, H, W]
    return x


def load_model(model, checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SpectroMaskLightningModule.load_from_checkpoint(
        checkpoint_path, model=model
    )
    model.to(device)
    model.eval()
    return model


def get_mask_from_array(
    audio_array: list[NDArray[np.float64]],
    model: SpectroMaskLightningModule,
    parameters: Parameters,
    debug=False,
) -> NDArray[np.uint8]:
    audio, network = parameters.audio, parameters.network

    is_phase_aware = model.model.n_channels == parameters.array.num_mics**2
    features, frequencies, times = build_spectrogram_features(
        audio_array, parameters, phase_aware=is_phase_aware
    )
    n_freqs, n_times = features.shape[-2:]

    features = _center_pad(features, parameters.network.image_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = preprocess_torch(features, device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)  # binary/multilabel
        masks = probs > 0.5

    if debug:
        print("logits:", logits)
        print("probs:", probs)
        print("prediction:", probs > 0.5)

    masks_np = masks.squeeze(1).cpu().numpy()

    diffX = network.image_size - n_times
    diffY = network.image_size - n_freqs

    start_Y = diffY // 2
    end_Y = start_Y + n_freqs

    start_X = diffX // 2
    end_X = start_X + n_times

    masks_np = masks_np[:, start_Y:end_Y, start_X:end_X]
    if debug:
        plot_spectrogram_4D(
            audio_array,
            audio.sampling_rate,
            parameters,
            mask=masks_np[0],
            is_db=False,
        )
    return masks_np[0]


def get_mask_from_array_arbitrary_size(
    audio_array: list[NDArray[np.float64]],
    model: SpectroMaskLightningModule,
    parameters: Parameters,
    chunk_size: int,
    chunk_overlap: int,
    debug=False,
) -> NDArray[np.bool_]:
    stft, network = parameters.stft, parameters.network
    
    N = audio_array.shape[1]

    if N < stft.n_fft:
        raise ValueError(f"Audio requires at least {stft.n_fft} samples; got {N}")

    if not 1 <= chunk_size <= network.image_size:
        raise ValueError(
            f"chunk_size must be between 1 and {network.image_size}; got {chunk_size}"
        )

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size")

    n_times = int(1 + np.floor((N - stft.n_fft) / stft.hop_length))

    if n_times <= chunk_size:
        return get_mask_from_array(audio_array, model, parameters, debug=debug)

    stride_frames = chunk_size - chunk_overlap
    chunk_samples = stft.n_fft + (chunk_size - 1) * stft.hop_length

    n_chunks = int(1 + np.ceil((n_times - chunk_size) / stride_frames))

    masks = None
    for i in range(n_chunks):
        start_frame = i * stride_frames
        start_sample = start_frame * stft.hop_length
        end_sample = min(N, start_sample + chunk_samples)

        chunk = audio_array[:, start_sample:end_sample]
        mask = get_mask_from_array(chunk, model, parameters, debug=debug)

        if masks is None:
            masks = mask
            continue

        actual_overlap = masks.shape[1] - start_frame

        if actual_overlap > 0:
            masks[:, -actual_overlap:] |= mask[:, :actual_overlap]
            mask = mask[:, actual_overlap:]

        masks = np.concatenate((masks, mask), axis=1)
    return masks[:, :n_times]


def main():
    args = parse_args()

    wav_path = args.wav_path
    parameters = Parameters.from_json(args.config)

    M = parameters.array.num_mics
    image_size = parameters.network.image_size

    if args.phase_aware:
        model = SpectroMaskNet(n_channels=M*M)
    else:
        model = SpectroMaskNet(n_channels=M)
    model = load_model(model, parameters.network.checkpoint_path)

    audio_array, sampling_rate = read_wav_file(wav_path, num_canals=M)

    if sampling_rate != parameters.audio.sampling_rate:
        raise ValueError(f"Invalid sampling_rate does not match parameters sampling_rate")
    mask = get_mask_from_array_arbitrary_size(
        audio_array=audio_array,
        model=model,
        parameters=parameters,
        chunk_size=image_size,
        chunk_overlap=image_size // 2,
    )
    plot_mask(mask)

    audio_array = bandpass_filter(audio_array, parameters.audio.sampling_rate, parameters.audio.min_freq, parameters.audio.max_freq)

    audio_array[:, :1000] = 0
    plot_waveform_4D(audio_array, parameters.audio.sampling_rate)

    plot_spectrogram_4D(audio_array, parameters.audio.sampling_rate, parameters, mask=mask, is_db=False)
    plot_spectrogram_4D(audio_array, parameters.audio.sampling_rate, parameters, is_db=True)


def crop_audio_array(
    audio_array: NDArray[np.float64], current_length: int, target_length: int
) -> NDArray[np.float64]:
    start = (current_length - target_length) // 2
    end = start + target_length
    return audio_array[:, start:end]


def pad_audio_array(
    audio_array: NDArray[np.float64], current_length: int, target_length: int
) -> NDArray[np.float64]:
    pad_width = target_length - current_length
    left = pad_width // 2
    right = pad_width - left
    return np.pad(audio_array, ((0, 0), (left, right)))


def crop_audio_canal(
    audio_canal: NDArray[np.float64], current_length: int, target_length: int
) -> NDArray[np.float64]:
    start = (current_length - target_length) // 2
    end = start + target_length
    return audio_canal[start:end]


def pad_audio_canal(
    audio_canal: NDArray[np.float64], current_length: int, target_length: int
) -> NDArray[np.float64]:
    pad_width = target_length - current_length
    left = pad_width // 2
    right = pad_width - left
    return np.pad(audio_canal, (left, right))


def pad_crop_audio_canal(audio_canal: NDArray[np.float64], sampling_rate, duration):
    target_length = int(sampling_rate * duration)
    current_length = len(audio_canal)

    if current_length > target_length:
        audio_canal = crop_audio_canal(audio_canal, current_length, target_length)
    elif current_length < target_length:
        audio_canal = pad_audio_canal(audio_canal, current_length, target_length)

    return audio_canal.copy()


def pad_crop_audio_array(audio_array: NDArray[np.float64], sampling_rate, duration):
    target_length = int(sampling_rate * duration)
    current_length = len(audio_array[0])
    if current_length > target_length:
        audio_array = crop_audio_array(audio_array, current_length, target_length)
    elif current_length < target_length:
        audio_array = pad_audio_array(audio_array, current_length, target_length)

    return audio_array.copy()


if __name__ == "__main__":
    main()
