import torch 
import torch.nn as nn
from .blocks import ConvLayerNormBlock

class AudioMaskNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        hidden_channels=32,
        kernel_size=15,
        output_logits=True,
        negative_slope=0.1,
    ):
        super().__init__()
        self.output_logits = output_logits
        self.net = nn.Sequential(
            ConvLayerNormBlock(in_channels, hidden_channels, kernel_size, negative_slope),
            ConvLayerNormBlock(hidden_channels, hidden_channels, kernel_size, negative_slope),
            ConvLayerNormBlock(hidden_channels, hidden_channels, kernel_size, negative_slope),
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)

        x = self.net(x)
        if not self.output_logits:
            x = torch.sigmoid(x)
        return x