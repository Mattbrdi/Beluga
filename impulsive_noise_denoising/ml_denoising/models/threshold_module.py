from pathlib import Path
import csv

import numpy as np 
import torch
import lightning as L

from metrics import compute_metrics
from thresholding import threshold_model
from impulsive_noise_denoising.ml_denoising.data_acquisition.data_parser import bandpass_waveform

class ThresholdMaskNet(L.LightningModule):
    def __init__(
        self,
        pulse_duration=0.005,
        pulse_overlap=0.0025,
        z_threshold=3.0,
        local_pulse_radius=10,
        max_plot_examples=8,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.plot_examples = []
        self.test_rows = []

    def forward(self, signal, frame_rate):
        predictions = []
        for sample, sample_rate in zip(signal.detach().cpu().numpy(), frame_rate.detach().cpu().numpy()):
            mask = threshold_model(
                sample,
                sample_rate,
                pulse_duration=0.0005,
                pulse_overlap=0.00025,
                z_threshold=10,
                local_pulse_radius=10,
            )
            channel_low_freq = bandpass_waveform(sample * sample, sample_rate, 1, 1000, 4)
            mask_low_freq = threshold_model(
                channel_low_freq,
                sample_rate,
                pulse_duration=0.008,
                pulse_overlap=0.0025,
                z_threshold=10,
                local_pulse_radius=5,
            )
            mask = (mask.astype(bool) | mask_low_freq.astype(bool)).astype(np.uint8)
            predictions.append(mask)
        return torch.as_tensor(np.stack(predictions), device=signal.device)

    def on_test_epoch_start(self):
        self.plot_examples.clear()
        self.test_rows.clear()

    def test_step(self, batch, batch_idx):
        signal = batch["input_signal"]
        target = batch["impulsive_mask"]
        frame_rate = batch["frame_rate"]
        pred = self(signal, frame_rate)

        metrics = compute_metrics(pred, target, frame_rate)
        self.log_dict(
            {
                "test_precision": metrics["test_precision"],
                "test_recall": metrics["test_recall"],
                "test_f1": metrics["test_f1"],
                "test_iou": metrics["test_iou"],
            },
            on_step=False,
            on_epoch=True,
            batch_size=target.shape[0],
        )
        self.log("test_fp_duration", metrics["test_fp_duration"], on_step=False, on_epoch=True, reduce_fx="sum")
        self.log("test_mi_duration", metrics["test_mi_duration"], on_step=False, on_epoch=True, reduce_fx="sum")
        self._collect_examples(batch, pred)
        self._collect_rows(batch, pred)
        return metrics

    def on_test_epoch_end(self):
        if not self.logger:
            return

        output_path = Path(self.logger.log_dir) / "per_segment_metrics.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "segment_name",
                    "test_precision",
                    "test_recall",
                    "test_f1",
                    "test_iou",
                    "test_fp_duration",
                    "test_mi_duration",
                ],
            )
            writer.writeheader()
            writer.writerows(self.test_rows)

    def _collect_examples(self, batch, pred):
        remaining = self.hparams.max_plot_examples - len(self.plot_examples)
        if remaining <= 0:
            return

        for index in range(min(remaining, pred.shape[0])):
            self.plot_examples.append(
                {
                    "signal": batch["input_signal"][index].detach().cpu(),
                    "target": batch["impulsive_mask"][index].detach().cpu(),
                    "pred": pred[index].detach().cpu(),
                    "frame_rate": int(batch["frame_rate"][index].detach().cpu().item()),
                    "segment_name": batch["segment_name"][index],
                }
            )

    def _collect_rows(self, batch, pred):
        for index in range(pred.shape[0]):
            metrics = compute_metrics(
                pred[index],
                batch["impulsive_mask"][index],
                int(batch["frame_rate"][index].detach().cpu().item()),
            )
            row = {
                "segment_name": batch["segment_name"][index],
                **{name: float(value.detach().cpu().item()) for name, value in metrics.items()},
            }
            self.test_rows.append(row)
