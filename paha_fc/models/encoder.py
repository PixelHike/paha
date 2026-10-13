"""Probabilistic encoder.

A ResNet-18 trunk with global average pooling feeds two linear heads that
parameterise the diagonal Gaussian posterior

    q_phi(z | x) = N(z; mu_phi(x), diag(exp(s_phi(x)))),

so the encoder parameters ``phi`` cover the trunk and both heads. The mean head
``mu_phi`` is the representation used for classification, centroid alignment,
latent memory and distillation; sampling is confined to reconstruction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18

FEATURE_DIM = 512


class Encoder(nn.Module):
    """Shared trunk + mean head + log-variance head.

    The first convolution is replaced by a 3x3 stride-1 block and the max
    pooling is removed, which preserves the spatial detail of 32-64 px images.
    """

    def __init__(self, latent_dim: int, pretrained: bool = False):
        super().__init__()
        backbone = resnet18(weights=None if not pretrained else "DEFAULT")
        if not pretrained:
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            backbone.maxpool = nn.Identity()
        self.trunk = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu,
                                   backbone.layer1, backbone.layer2, backbone.layer3,
                                   backbone.layer4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head_mu = nn.Linear(backbone.fc.in_features, latent_dim)
        self.head_logvar = nn.Linear(backbone.fc.in_features, latent_dim)
        # A near-deterministic posterior at initialisation: with unit sampling
        # variance the reconstruction gradient is dominated by noise, so the
        # log-variance head starts at exp(-5).
        nn.init.zeros_(self.head_logvar.weight)
        nn.init.constant_(self.head_logvar.bias, -5.0)

    def forward(self, x: torch.Tensor):
        h = self.avgpool(self.trunk(x))
        h = torch.flatten(h, 1)
        return self.head_mu(h), self.head_logvar(h)

    @torch.no_grad()
    def mean_embedding(self, x: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        """Chunked ``mu_phi`` evaluation, used for centroids and evaluation."""
        outs = []
        for i in range(0, x.shape[0], chunk):
            outs.append(self.forward(x[i:i + chunk])[0])
        return torch.cat(outs) if outs else x.new_empty((0, self.head_mu.out_features))
