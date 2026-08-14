import numpy as np
from numpy.typing import NDArray


import torch 
import argparse
from torchvision import transforms


import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from time_frequency_mask.configuration import SAMPLING_RATE, N_FFT, HOP_LENGTH, N_TIMES, N_FREQS, IMAGE_SIZE, DURATION, CKPT_PATH
from time_frequency_mask.plotter import plot_spectrogram_4D, plot_mask,plot_waveform_4D
from time_frequency_mask.stft import frequency_band, scipy_spectrogram, scipy_db_spectrogram

from time_frequency_mask.masknet.models.spectro_mask_net_lightning import SpectroMaskLightningModule
from time_frequency_mask.masknet.models.spectro_mask_net import SpectroMaskNet

from time_frequency_mask.data_generation.models.mask import AudioMask
from time_frequency_mask.data_generation.core.preprocess import bandpass_filter
from time_frequency_mask.data_generation.io.data_parser import read_wav_file

# CKPT_PATH = r"C:\Users\amine\Desktop\Canada\Beluga\runs\spectro_mask_net\version_12\checkpoints\epoch=111-step=3584.ckpt"
# CKPT_PATH = r"C:\Users\amine\Desktop\Canada\Beluga\runs\spectro_mask_net\version_16\epoch=155-step=17316.ckpt"
# CKPT_PATH = r"C:\Users\amine\Desktop\Canada\Beluga\best-epoch=157-val_loss=0.2527.ckpt"
# CKPT_PATH = r"C:\Users\amine\Desktop\Canada\Beluga\last.ckpt"
WAV_PATH = r"C:\Users\amine\Desktop\Canada\Beluga\time_frequency_mask\data_generation\data\input\beluga_2026_test_duration_1_1s.wav"#r"C:\Users\amine\Downloads\amine.wav"#
# WAV_PATH = r"C:\Users\amine\Desktop\Canada\Beluga\time_frequency_mask\data_generation\data\input\beluga_synth_02.wav"#r"C:\Users\amine\Downloads\amine.wav"#
WAV_PATH = r"C:\Users\amine\Desktop\Canada\Beluga\pipeline\test_data2026_all\data\beluga_2026_test_duration_5_4s.wav"
is_db = False

def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic beluga mask generator")
    parser.add_argument("--checkpoint-path", help="Provide checkpoint path",default=CKPT_PATH, type=str)
    parser.add_argument("--wav-path" , help="provide wav path", default=WAV_PATH, type=str)

    return parser.parse_args()

def preprocess_waveform(waveform : NDArray[np.float64]):
    waveform = bandpass_filter(waveform, SAMPLING_RATE)
    scale = np.percentile(np.abs(waveform), 95)
    if scale == 0:
        scale = 1
    waveform = waveform / scale
    return waveform

# def compute_stft(waveform: NDArray[np.float64]):
#     freqs, times, Zxx = scipy_stft(waveform, SAMPLING_RATE, n_fft = N_FFT, hop_length = HOP_LENGTH)

#     freqs, Zxx = frequency_band(freqs, Zxx)

#     if Zxx.shape[0] != N_FREQS or Zxx.shape[1] != N_TIMES:
#         raise ValueError(f"incorrect shape got {Zxx.shape} instead of {(N_FREQS, N_TIMES)}")

#     return freqs, times, Zxx

# def compute_log_stft(waveform: NDArray[np.float64]):
#     freqs, times, Zxx = compute_stft(waveform)
#     return freqs, times, 20 * np.log10(np.abs(np.maximum(2 * Zxx, 1e-12)))

def preprocess_torch(Ds : NDArray[np.float64], device):    
    # transform = transforms.Compose([
    #     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),   # use same size as training
    #     # transforms.Normalize(mean=[...], std=[...]),  # use same normalization as training
    # ])

    Ds = torch.from_numpy(Ds).float()
    x = Ds.unsqueeze(0).to(device)  # shape: [1, C, H, W]
    return x 

def load_model(checkpoint_path = CKPT_PATH):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = SpectroMaskLightningModule.load_from_checkpoint(checkpoint_path, model=SpectroMaskNet())
    model.to(device)
    model.eval()
    return model

def get_mask_from_array(audio_array : list[NDArray[np.float64]], model, debug=False) -> NDArray[np.uint8]:
    N = audio_array.shape[1]

    Ds = []

    freqs_list = []

    times_list = []

    for canal in audio_array:
        # canal = preprocess_waveform(canal)

        if is_db:
            freqs, times, D = scipy_db_spectrogram(canal)
        else:
            freqs, times, D = scipy_spectrogram(canal)
        
        freqs, D = frequency_band(freqs, D)

        D = np.abs(D)
        if np.max(np.abs(D)) > 0:
            D = D - np.min(D)
            D = D / np.percentile(np.abs(D),99)
            D = np.clip(D, 0, 1)

        # D = np.clip(D, -1,1)

        n_freqs, n_times = D.shape
        if D.shape[0] > IMAGE_SIZE or D.shape[1] > IMAGE_SIZE:
            raise ValueError(f"incorrect shape got {D.shape} greater or equal than {(IMAGE_SIZE, IMAGE_SIZE)}")

        diffX = IMAGE_SIZE - n_times 
        diffY = IMAGE_SIZE - n_freqs

        if diffX < 0 or diffY < 0:
            raise ValueError(f"D shape {D.shape} is larger than IMAGE_SIZE {IMAGE_SIZE}")

        D = np.pad(D, ((diffY // 2, diffY - diffY // 2), (diffX // 2, diffX - diffX // 2)))

        Ds.append(D)
        freqs_list.append(freqs)
        times_list.append(times)

    Ds = np.stack(Ds, axis=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = preprocess_torch(Ds, device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)  # binary/multilabel
        masks = probs > 0.5

    if debug:
        print("logits:", logits)
        print("probs:", probs)
        print("prediction:", probs > 0.5)

    masks_np = masks.squeeze(1).cpu().numpy()
    diffX = IMAGE_SIZE - n_times
    diffY = IMAGE_SIZE - n_freqs

    start_Y = diffY // 2
    end_Y = start_Y + n_freqs

    start_X = diffX // 2
    end_X = start_X + n_times

    masks_np = masks_np[:, start_Y:end_Y, start_X: end_X]
    if debug:
        plot_spectrogram_4D(audio_array, SAMPLING_RATE, is_db=False, mask=AudioMask(masks_np[0], masks_np.shape[0], masks_np[1], SAMPLING_RATE).data)
    return masks_np[0]

def get_mask_from_array_arbitrary_size(audio_array : list[NDArray[np.float64]], model, chunk_size = IMAGE_SIZE, chunk_overlap = IMAGE_SIZE // 2, debug=False) -> NDArray[np.uint8]:
    N = audio_array.shape[1]

    if N < N_FFT:
        raise ValueError(f"Audio requires at least {N_FFT} samples; got {N}")

    if not 1 <= chunk_size <= IMAGE_SIZE:
        raise ValueError(
            f"chunk_size must be between 1 and {IMAGE_SIZE}; got {chunk_size}"
        )

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size"
        )
    
    n_times = int(1 + np.floor((N - N_FFT) / HOP_LENGTH))

    if n_times <= chunk_size:
        return get_mask_from_array(audio_array, model, debug=debug)
    
    stride_frames = chunk_size - chunk_overlap
    chunk_samples = N_FFT + (chunk_size - 1) * HOP_LENGTH

    n_chunks = int(1 + np.ceil((n_times - chunk_size) / stride_frames))

    masks = None
    for i in range(n_chunks):
        start_frame = i*stride_frames
        start_sample = start_frame * HOP_LENGTH
        end_sample = min(N, start_sample + chunk_samples)

        chunk = audio_array[:, start_sample:end_sample]
        mask = get_mask_from_array(chunk, model, debug=debug)

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

    checkpoint_path = args.checkpoint_path
    wav_path = args.wav_path

    model = load_model()

    audio_array, sampling_rate = read_wav_file(wav_path)
    # N = (HOP_LENGTH*255) + N_FFT    
    # print(N/SAMPLING_RATE)
    # # audio_array = audio_array[:, :N]
    mask = get_mask_from_array_arbitrary_size(audio_array, model)
    plot_mask(mask)

    audio_array = bandpass_filter(audio_array, SAMPLING_RATE)

    audio_array[:,:1000] = 0
    plot_waveform_4D(audio_array, SAMPLING_RATE)

    plot_spectrogram_4D(audio_array, SAMPLING_RATE, is_db=False, mask=mask)
    plot_spectrogram_4D(audio_array, SAMPLING_RATE, is_db=True, fmin=100)

def crop_audio_array(audio_array : NDArray[np.float64], current_length : int, target_length : int) -> NDArray[np.float64]:
    start = (current_length - target_length) // 2
    end = start + target_length
    return audio_array[:, start:end] 

def pad_audio_array(audio_array : NDArray[np.float64], current_length : int, target_length : int) -> NDArray[np.float64]:
    pad_width = target_length - current_length
    left = pad_width // 2
    right = pad_width - left
    return np.pad(audio_array, ((0,0), (left, right)))

def crop_audio_canal(audio_canal : NDArray[np.float64], current_length : int, target_length : int) -> NDArray[np.float64]:
    start = (current_length - target_length) // 2
    end = start + target_length
    return audio_canal[start:end]

def pad_audio_canal(audio_canal : NDArray[np.float64], current_length : int, target_length : int) -> NDArray[np.float64]:
    pad_width = target_length - current_length
    left = pad_width // 2
    right = pad_width - left
    return np.pad(audio_canal, (left, right))

def pad_crop_audio_canal(audio_canal : NDArray[np.float64]):
    target_length = int(SAMPLING_RATE * DURATION)
    current_length = len(audio_canal)

    if current_length > target_length: 
        audio_canal = crop_audio_canal(audio_canal, current_length, target_length)
    elif current_length < target_length:
        audio_canal = pad_audio_canal(audio_canal, current_length, target_length)
    
    return audio_canal.copy()

def pad_crop_audio_array(audio_array : NDArray[np.float64]):
    target_length = int(SAMPLING_RATE * DURATION)
    current_length = len(audio_array[0])
    if current_length > target_length: 
        audio_array = crop_audio_array(audio_array, current_length, target_length)
    elif current_length < target_length:
        audio_array = pad_audio_array(audio_array, current_length, target_length)
    
    return audio_array.copy()

if __name__ == '__main__':
    main()