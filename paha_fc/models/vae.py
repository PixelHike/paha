"""Variational pieces and the joint parameter vector ``W = [theta_phi; theta_g]``."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

ENC_PREFIX = "enc."
DEC_PREFIX = "dec."


def reparameterize(mu: torch.Tensor, log_s2: torch.Tensor) -> torch.Tensor:
    """One reparameterised posterior sample: ``z = mu + exp(s/2) * eps``."""
    std = torch.exp(0.5 * log_s2.clamp(min=-10.0))
    return mu + std * torch.randn_like(std)


def gaussian_kl(mu: torch.Tensor, log_s2: torch.Tensor) -> torch.Tensor:
    """Analytic diagonal-Gaussian KL term against a standard normal prior, per sample."""
    return 0.5 * (mu.pow(2) + log_s2.exp() - 1.0 - log_s2).sum(dim=1)


def pack_state(enc: nn.Module, dec: nn.Module,
               device: torch.device = None) -> Dict[str, torch.Tensor]:
    """Flatten encoder and decoder parameters into one ``W`` dictionary."""
    state = {ENC_PREFIX + k: v.detach().clone() for k, v in enc.state_dict().items()}
    state.update({DEC_PREFIX + k: v.detach().clone() for k, v in dec.state_dict().items()})
    if device is not None:
        state = {k: v.to(device) for k, v in state.items()}
    return state


def unpack_state(state: Dict[str, torch.Tensor], enc: nn.Module, dec: nn.Module) -> None:
    enc.load_state_dict({k[len(ENC_PREFIX):]: v for k, v in state.items()
                         if k.startswith(ENC_PREFIX)})
    dec.load_state_dict({k[len(DEC_PREFIX):]: v for k, v in state.items()
                         if k.startswith(DEC_PREFIX)})


def state_norm(state: Dict[str, torch.Tensor]) -> float:
    return float(sum(float(v.pow(2).sum()) for v in state.values()) ** 0.5)


class AutoEncoder(nn.Module):
    """Trainable encoder-decoder pair addressed as a single parameter vector."""

    def __init__(self, enc: nn.Module, dec: nn.Module):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.enc(x)

    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)[0]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def reconstruct(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_s2 = self.enc(x)
        z = reparameterize(mu, log_s2)
        return self.dec(z), mu, log_s2

    def state(self, device: torch.device = None) -> Dict[str, torch.Tensor]:
        return pack_state(self.enc, self.dec, device)

    def set_state(self, state: Dict[str, torch.Tensor]) -> None:
        unpack_state(state, self.enc, self.dec)

    def freeze(self) -> "AutoEncoder":
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self


def build_autoencoder(latent_dim: int, input_size: int, device=None,
                      pretrained: bool = False) -> AutoEncoder:
    from .decoder import Decoder
    from .encoder import Encoder
    enc = Encoder(latent_dim=latent_dim, pretrained=pretrained)
    dec = Decoder(latent_dim=latent_dim, input_size=input_size)
    ae = AutoEncoder(enc, dec)
    if device is not None:
        ae.to(device)
    return ae


@torch.no_grad()
def anchor_forward(anchor: AutoEncoder, x: torch.Tensor, chunk: int = 2048
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cached task-anchor outputs ``mu_init(x)`` and ``g_init(mu_init(x))``."""
    mus, recs = [], []
    for i in range(0, x.shape[0], chunk):
        mu = anchor.enc(x[i:i + chunk])[0]
        mus.append(mu)
        recs.append(anchor.dec(mu))
    if not mus:
        empty = x.new_empty((0,))
        return empty, empty
    return torch.cat(mus), torch.cat(recs)
