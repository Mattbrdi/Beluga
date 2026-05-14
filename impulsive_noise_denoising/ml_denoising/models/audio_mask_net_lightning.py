import torch
import torch.nn as nn
import lightning as L

from metrics import compute_metrics

class AudioMaskLightningModule(L.LightningModule):
    def __init__(self, model, lr=1e-3, pos_weight=torch.tensor([20.0])):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch["input_signal"].float()
        y = batch["impulsive_mask"].float()

        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["input_signal"].float()
        y = batch["impulsive_mask"].float()
        frame_rate = batch["frame_rate"]

        logits = self(x)
        loss = self.loss_fn(logits, y)

        pred = torch.sigmoid(logits) > 0.5
        metrics = compute_metrics(pred, y, frame_rate)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        prob = torch.sigmoid(logits)
        self.log("val_prob_mean", prob.mean(), prog_bar=True)
        self.log("val_prob_max", prob.max(), prog_bar=True)
        
        self.log_dict(
            {
                "val_iou": metrics["test_iou"],
                "val_f1": metrics["test_f1"],
                "val_precision": metrics["test_precision"],
                "val_recall": metrics["test_recall"],
            },
            prog_bar=True,
            on_epoch=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        x = batch["input_signal"].float()
        y = batch["impulsive_mask"].float()
        frame_rate = batch["frame_rate"]

        logits = self(x)
        pred = torch.sigmoid(logits) > 0.5

        metrics = compute_metrics(pred, y, frame_rate)
        self.log_dict(metrics, on_epoch=True)

        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
