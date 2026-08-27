import argparse

from time_frequency_mask.config import Parameters
from time_frequency_mask.masknet.dataset import WhistleMaskDataModule
from time_frequency_mask.masknet.models.spectro_mask_net_lightning import SpectroMaskLightningModule
from time_frequency_mask.masknet.models.spectro_mask_net import SpectroMaskNet
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

import lightning as L


def parse_args():
    parser = argparse.ArgumentParser(description="Train the whistle TF-mask U-Net.")
    parser.add_argument(
        "--phase-aware",
        action="store_true",
        help=(
            "Use M*M input channels: four magnitudes and real/imaginary IPD "
            "features for all M(M-1) microphone pairs."
        ),
    )
    parser.add_argument("--config", help="Configuration used for time_frequency_mask", required=True, type=str)
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to resume training ")
    return parser.parse_args()


def main():
    args = parse_args()

    parameters = Parameters.from_json(args.config)
    M = parameters.array.num_mics
    datamodule = WhistleMaskDataModule(
        parameters=parameters,
        dataset_path=parameters.network.input_path,
        batch_size=24,
        seed=42,
        num_workers=8,
        phase_aware=args.phase_aware,
    )

    model = SpectroMaskLightningModule(
        model=SpectroMaskNet(
            n_channels=M*M if args.phase_aware else M,
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
