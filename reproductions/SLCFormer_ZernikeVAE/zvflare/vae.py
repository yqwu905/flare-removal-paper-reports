"""ZernikeVAE: stochastic refinement of the physically-rendered scatter flare.

From the paper:

    "After simulation, our pipeline adopts an encoder-decoder architecture to
     refine the generated flare.  The encoder processes the concatenated Zernike
     coefficients and kernel size map to estimate the latent distribution's mean
     mu and log-variance log sigma^2.  A latent code z is then sampled via the
     reparameterization trick to model stochastic optical distortions.  The
     decoder reconstructs the final ZernikeVAE flare."

The paper gives no architecture, latent dimension, or supervision, so this is a
faithful, fully-specified realisation (documented in the README).  Because the
task is to *refine* a flare that is already rendered (not hallucinate one from
scratch), the model is a **conditional VAE-UNet**:

* **Input descriptor** (per pixel): the physics-rendered flare RGB, a handful of
  dominant Zernike coefficient maps, and the local kernel-size map.
* **Encoder** with skip connections -> bottleneck;  a global head produces
  (mu, logvar);  reparameterisation -> z.
* **z modulates the bottleneck** (injecting stochastic optical variation), the
  **skips carry the rendered-flare structure** (so the output stays sharp), and
  the **decoder** outputs the refined flare RGB.
* **Loss**: reconstruction (L1 + L2) to the rendered flare + beta * KL.  At
  sample time, drawing z from the prior (with a temperature) yields diverse
  plausible flares -- the role the paper assigns to the reparameterisation trick.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["VAEConfig", "ZernikeVAE", "vae_loss"]


@dataclass
class VAEConfig:
    in_channels: int = 8      # 3 flare RGB + 4 Zernike maps + 1 kernel-size map
    out_channels: int = 3
    base: int = 32
    latent_dim: int = 128
    img_size: int = 128
    depth: int = 4            # number of stride-2 stages
    kl_weight: float = 1e-3


def _block(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(inplace=True),
    )


def _plain(cin: int, cout: int) -> nn.Sequential:
    """Stride-1 fuse block used in the decoder (after concatenating a skip)."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(inplace=True),
    )


class ZernikeVAE(nn.Module):
    def __init__(self, cfg: VAEConfig | None = None):
        super().__init__()
        self.cfg = cfg or VAEConfig()
        c = self.cfg
        chans = [c.base * (2 ** i) for i in range(c.depth)]  # per-stage widths
        self.stem = nn.Conv2d(c.in_channels, c.base, 3, padding=1)
        # ---- encoder (collect skips before each downsample) ----
        self.down = nn.ModuleList()
        prev = c.base
        for ch in chans:
            self.down.append(_block(prev, ch, stride=2))
            prev = ch
        self.feat_size = c.img_size // (2 ** c.depth)
        feat_dim = prev * self.feat_size * self.feat_size
        # ---- latent head + projection back into the bottleneck ----
        self.fc_mu = nn.Linear(feat_dim, c.latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, c.latent_dim)
        self.fc_dec = nn.Linear(c.latent_dim, prev * self.feat_size * self.feat_size)
        self._bottleneck_ch = prev
        # ---- decoder (concat skips) ----
        self.up = nn.ModuleList()
        rev = list(reversed(chans))               # bottleneck -> ... widths
        skip_ch = [c.base] + chans[:-1]           # skip widths captured by stem/down
        skip_ch = list(reversed(skip_ch))
        cur = prev
        for i, ch in enumerate(rev):
            out_ch = skip_ch[i]
            self.up.append(nn.ModuleList([
                nn.Upsample(scale_factor=2, mode="nearest"),
                _plain(cur + out_ch, out_ch),       # concat skip then fuse
            ]))
            cur = out_ch
        self.head = nn.Conv2d(cur, c.out_channels, 3, padding=1)

    # ------------------------------------------------------------- encode
    def encode(self, x: torch.Tensor):
        s = self.stem(x)
        skips = [s]
        h = s
        for blk in self.down:
            h = blk(h)
            skips.append(h)
        bottleneck = skips.pop()                    # deepest feature
        flat = bottleneck.flatten(1)
        return self.fc_mu(flat), self.fc_logvar(flat), skips, bottleneck

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    # ------------------------------------------------------------- decode
    def decode(self, z: torch.Tensor, skips, bottleneck) -> torch.Tensor:
        zb = self.fc_dec(z).view(-1, self._bottleneck_ch, self.feat_size, self.feat_size)
        h = bottleneck + zb                         # z modulates the bottleneck
        for (up, fuse), skip in zip(self.up, reversed(skips)):
            h = up(h)
            h = fuse(torch.cat([h, skip], dim=1))
        return F.softplus(self.head(h))             # non-negative linear radiance

    def forward(self, x: torch.Tensor):
        mu, logvar, skips, bottleneck = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, skips, bottleneck), mu, logvar

    @torch.no_grad()
    def sample_refinement(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Draw a stochastic refined flare (temperature scales the latent noise)."""
        mu, logvar, skips, bottleneck = self.encode(x)
        z = mu + temperature * torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z, skips, bottleneck)


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float,
) -> tuple[torch.Tensor, dict]:
    rec = F.l1_loss(recon, target) + F.mse_loss(recon, target)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = rec + kl_weight * kl
    return loss, {"rec": float(rec.detach()), "kl": float(kl.detach()), "loss": float(loss.detach())}
