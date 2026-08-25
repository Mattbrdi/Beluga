import argparse
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from time_frequency_mask.masknet.dataset import WhistleMaskDataModule
from time_frequency_mask.masknet.models.spectro_mask_net_lightning import SpectroMaskLightningModule
from time_frequency_mask.masknet.models.spectro_mask_net import SpectroMaskNet
from time_frequency_mask.configuration import DATASET_PATH
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

import lightning as L


def parse_args():
    parser = argparse.ArgumentParser(description="Train the whistle TF-mask U-Net.")
    parser.add_argument(
        "--phase-aware",
        action="store_true",
        help=(
            "Use 16 input channels: four magnitudes and real/imaginary IPD "
            "features for all six microphone pairs."
        ),
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to resume training ")
    return parser.parse_args()


def main():
    args = parse_args()
    datamodule = WhistleMaskDataModule(
        dataset_path=DATASET_PATH,
        batch_size=24,
        seed=42,
        num_workers=8,
        phase_aware=args.phase_aware,
    )

    model = SpectroMaskLightningModule(
        model=SpectroMaskNet(
            n_channels=16 if args.phase_aware else 4,
            dropout=0.1,
        ),
        lr=1e-3,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best-{epoch:03d}-{val_loss:.4f}",
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=30,
        min_delta=1e-4,
    )

    trainer = L.Trainer(
        max_epochs=1000,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=CSVLogger("runs", name="spectro_mask_net")
    )

    
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.checkpoint)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
