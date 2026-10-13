"""Decoder ``g : R^m -> R^n`` mapping a latent code back to an image.

A linear map reshapes the latent code into a 4x4 feature map that a chain of
transposed-convolution blocks doubles up to the target resolution. The final
convolution is followed by ``tanh`` because images are normalised to [-1, 1].
"""

from __future__ import annotations

import torch
import torch.nn as nn

BASE = 4
START_CHANNELS = 512
MIN_CHANNELS = 16


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, input_size: int, in_channels: int = 3):
        super().__init__()
        size, blocks_needed = BASE, 0
        while size < input_size:
            size *= 2
            blocks_needed += 1
        if size != input_size:
            raise ValueError(f"input_size must be {BASE}*2^k, got {input_size}")
        self.input_size = int(input_size)
        self.base = BASE
        self.num_blocks = blocks_needed

        self.map = nn.Sequential(nn.Linear(latent_dim, START_CHANNELS * BASE * BASE),
                                 nn.ReLU(inplace=True))
        channels = START_CHANNELS
        stages = []
        for _ in range(blocks_needed):
            out_c = max(channels // 2, MIN_CHANNELS)
            stages.append(nn.Sequential(
                nn.ConvTranspose2d(channels, out_c, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ))
            channels = out_c
        self.blocks = nn.Sequential(*stages)
        self.final = nn.Conv2d(channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.map(z)
        h = h.view(h.size(0), -1, self.base, self.base)
        return torch.tanh(self.final(self.blocks(h)))
