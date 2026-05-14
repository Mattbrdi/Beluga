import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import ImpulsiveNoiseDataModule
from models.audio_mask_net_lightning import AudioMaskLightningModule
from models.audio_mask_net import AudioMaskNet

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

import lightning as L

datamodule = ImpulsiveNoiseDataModule(
    dataset_path="impulsive_noise_denoising/ml_denoising/data/dataset_1/output",
    batch_size=4,
    seed=42,
)

model = AudioMaskLightningModule(
    model=AudioMaskNet(),
    lr=1e-3,
)

trainer = L.Trainer(
    max_epochs=20,
    accelerator="auto",
    devices="auto",
    logger=CSVLogger("runs", name="audio_mask_net"),
    callbacks=[
        ModelCheckpoint(monitor="val_f1", mode="max"),
        EarlyStopping(monitor="val_f1", mode="max", patience=5),
    ],
)

trainer.fit(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule)
