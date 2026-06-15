"""Train the ZernikeVAE that refines physics-rendered scatter flares.

Training pairs are generated on the fly by the synthesizer: the input is the
8-channel descriptor (flare RGB + dominant Zernike maps + kernel-size map) and
the reconstruction target is the rendered flare RGB.  The KL term injects the
stochastic optical variation sampled at generation time.

Usage:
    python scripts/train_vae.py --steps 400 --size 128 --out assets
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zvflare import FlareSynthesizer, SynthConfig            # noqa: E402
from zvflare.optics import OpticsConfig                       # noqa: E402
from zvflare.vae import VAEConfig, ZernikeVAE, vae_loss       # noqa: E402
from zvflare.utils import hstack, save_image                  # noqa: E402


def make_batch(synth: FlareSynthesizer, bs: int, size: int, rng):
    xs, ys = [], []
    for _ in range(bs):
        pair = synth.synthesize(rng)
        desc = synth.vae_descriptor(pair)                # (H,W,8) in [0,1]
        tgt = desc[..., :3]                              # rendered flare RGB (target)
        xs.append(desc.transpose(2, 0, 1))
        ys.append(tgt.transpose(2, 0, 1))
    x = torch.from_numpy(np.stack(xs)).float()
    y = torch.from_numpy(np.stack(ys)).float()
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kl", type=float, default=1e-3)
    ap.add_argument("--out", default="assets")
    ap.add_argument("--ckpt", default="assets/zernike_vae.pt")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    cfg = SynthConfig(image_size=args.size,
                      optics=OpticsConfig(pupil_size=80, psf_size=72, oversample=2,
                                          num_zernike=18, chromatic=True),
                      field_strength=0.5, n_light_sources=6)
    synth = FlareSynthesizer(cfg)
    rng = np.random.default_rng(0)

    vcfg = VAEConfig(in_channels=8, out_channels=3, base=24, latent_dim=96,
                     img_size=args.size, depth=4, kl_weight=args.kl)
    model = ZernikeVAE(vcfg)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    print("Training ZernikeVAE: %.2fM params, %d steps, batch %d, %dx%d"
          % (sum(p.numel() for p in model.parameters()) / 1e6, args.steps,
             args.batch, args.size, args.size))
    t0 = time.time()
    hist = []
    model.train()
    for step in range(1, args.steps + 1):
        x, y = make_batch(synth, args.batch, args.size, rng)
        recon, mu, logvar = model(x)
        loss, logs = vae_loss(recon, y, mu, logvar, vcfg.kl_weight)
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(logs)
        if step % max(1, args.steps // 10) == 0 or step == 1:
            print("  step %4d/%d  loss %.4f  rec %.4f  kl %.4f  (%.1fs)"
                  % (step, args.steps, logs["loss"], logs["rec"], logs["kl"], time.time() - t0))

    torch.save({"model": model.state_dict(), "vcfg": vcfg.__dict__}, args.ckpt)

    # ---- qualitative: input flare vs VAE reconstruction vs stochastic samples ----
    model.eval()
    x, y = make_batch(synth, 1, args.size, rng)
    with torch.no_grad():
        recon, _, _ = model(x)
        samples = [model.sample_refinement(x, temperature=1.2)[0] for _ in range(3)]

    def chw2hwc(t):
        a = t.detach().cpu().numpy().transpose(1, 2, 0)
        return a / max(a.max(), 1e-6)

    panel = hstack([chw2hwc(y[0]), chw2hwc(recon[0])] + [chw2hwc(s) for s in samples])
    save_image(os.path.join(args.out, "vae_refinement.png"), panel)

    print("Loss improved %.4f -> %.4f over %d steps"
          % (np.mean([h["loss"] for h in hist[:20]]),
             np.mean([h["loss"] for h in hist[-20:]]), args.steps))
    print("Saved checkpoint to", args.ckpt, "and vae_refinement.png to", args.out)


if __name__ == "__main__":
    main()
