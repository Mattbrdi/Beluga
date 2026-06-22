import torch
import torch.nn as nn
import lightning as L
from torchmetrics.segmentation import DiceScore
from time_frequency_mask.configuration import SAMPLING_RATE

def dice_loss_from_logits(logits, targets, eps=1e-6):
    targets = targets.float()
    probs = torch.sigmoid(logits)

    probs = probs.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)

    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + eps) / (denominator + eps)

    return 1 - dice.mean()

class SpectroMaskLightningModule(L.LightningModule):
    def __init__(self, model, lr=1e-3, pos_weight=torch.tensor([20.0])):
        super().__init__()
        self.model = model
        self.lr = lr
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice_score = DiceScore(num_classes=1)
        # self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight) + DiceScore(num_classes=1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch["stft"].float()
        y = batch["mask"].float()

        logits = self(x)
        targets = y.float()

        bce_loss = self.bce_loss(logits, targets)
        dice_loss = dice_loss_from_logits(logits, targets)

        loss = bce_loss + dice_loss

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["stft"].float()
        y = batch["mask"].float()
        frame_rate = SAMPLING_RATE

        logits = self(x)
        targets = y.float()

        bce_loss = self.bce_loss(logits, targets)
        dice_loss = dice_loss_from_logits(logits, targets)

        loss = bce_loss + dice_loss

        # pred = torch.sigmoid(logits) > 0.5
        # metrics = compute_metrics(pred, targets, frame_rate)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        prob = torch.sigmoid(logits)
        self.log("val_prob_mean", prob.mean(), prog_bar=True)
        self.log("val_prob_max", prob.max(), prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x = batch["stft"].float()
        y = batch["mask"].float()

        logits = self(x)
        targets = y.float()

        bce_loss = self.bce_loss(logits, targets)
        dice_loss = dice_loss_from_logits(logits, targets)

        loss = bce_loss + dice_loss

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
