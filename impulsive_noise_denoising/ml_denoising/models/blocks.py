import torch.nn as nn 

class ConvLayerNormBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, negative_slope=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.activation(x)