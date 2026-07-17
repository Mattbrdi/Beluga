import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
from scipy.io.wavfile import write

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from time_frequency_mask.data_generation.data_parser import read_wav_file

SAMPLE_RATE = 192000
CLIP_LENGTH = 2**16
INPUT_DIR = Path(__file__).resolve().parent / "data" / "input" / "whistle"
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "output" / "diffusion_sample.wav"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "data" / "output" / "diffusion_model.pt"
EPOCHS = 100
LR = 1e-4
BATCH_SIZE = 1
SAMPLE_STEPS = 100
CROPS_PER_EPOCH = 256
print(f"is cuda {torch.cuda.is_available()}")


def load_waveforms(input_dir: Path) -> list[torch.Tensor]:
    wav_paths = sorted(input_dir.glob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No WAV files found in {input_dir}")

    waveforms = []
    for wav_path in wav_paths:
        audio_array, sample_rate = read_wav_file(wav_path, num_canals=1)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sample_rate} Hz for {wav_path}")

        waveform = torch.from_numpy(audio_array[0]).float()
        waveform = waveform - waveform.mean()
        waveform = waveform / torch.clamp(waveform.abs().max(), min=1e-8)
        waveforms.append(waveform)

    return waveforms


class RandomCropDataset(Dataset):
    def __init__(self, waveforms: list[torch.Tensor], clip_length: int, crops_per_epoch: int):
        self.waveforms = waveforms
        self.clip_length = clip_length
        self.crops_per_epoch = crops_per_epoch

    def __len__(self) -> int:
        return self.crops_per_epoch

    def __getitem__(self, index: int) -> torch.Tensor:
        waveform = self.waveforms[torch.randint(len(self.waveforms), ()).item()]
        if waveform.numel() <= self.clip_length:
            crop = torch.nn.functional.pad(waveform, (0, self.clip_length - waveform.numel()))
        else:
            start = torch.randint(waveform.numel() - self.clip_length + 1, ()).item()
            crop = waveform[start:start + self.clip_length]

        return crop.unsqueeze(0)


model = DiffusionModel(
    net_t=UNetV0, # The model type used for diffusion (U-Net V0 in this case)
    in_channels=1, # U-Net: number of input/output (audio) channels
    channels=[8, 16, 32, 64, 128, 256], # U-Net: channels at each layer
    factors=[1, 4, 4, 4, 2, 2], # U-Net: downsampling and upsampling factors at each layer
    items=[1, 1, 1, 2, 2, 2], # U-Net: number of repeating items at each layer
    attentions=[0, 0, 0, 0, 1, 1], # U-Net: attention enabled/disabled at each layer
    attention_heads=4, # U-Net: number of attention heads per attention item
    attention_features=32, # U-Net: number of attention features per attention item
    diffusion_t=VDiffusion, # The diffusion method used
    sampler_t=VSampler, # The diffusion sampler used
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

if CHECKPOINT_PATH.exists():
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

waveforms = load_waveforms(INPUT_DIR)
dataset = RandomCropDataset(waveforms, CLIP_LENGTH, CROPS_PER_EPOCH)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            loss = model(batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {running_loss / len(dataloader):.6f}")

CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), CHECKPOINT_PATH)
print(f"Saved checkpoint to {CHECKPOINT_PATH}")

del waveforms, dataset, dataloader, optimizer
if device.type == "cuda":
    torch.cuda.empty_cache()

# Turn noise into new audio sample with diffusion
model.eval()
with torch.no_grad():
    noise = torch.randn(1, 1, CLIP_LENGTH, device=device) # [batch_size, in_channels, length]
    sample = model.sample(noise, num_steps=SAMPLE_STEPS) # Suggested num_steps 10-100

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sample_waveform = sample[0, 0].detach().cpu()
sample_waveform = sample_waveform / torch.clamp(sample_waveform.abs().max(), min=1e-8)
write(OUTPUT_PATH, SAMPLE_RATE, sample_waveform.numpy().astype("float32"))
print(f"Saved generated sample to {OUTPUT_PATH}")
