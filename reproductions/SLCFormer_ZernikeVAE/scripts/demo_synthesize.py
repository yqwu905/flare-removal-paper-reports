"""Demo: synthesize a scatter-flare training pair and reproduce the paper's
Fig.1 comparison (single scalar PSF a la Flare7K  vs.  spatially-varying PSF).

Usage:
    python scripts/demo_synthesize.py --out assets --seed 0 [--bg path/to/img.jpg]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zvflare import FlareSynthesizer, SynthConfig          # noqa: E402
from zvflare.optics import PSFGenerator                    # noqa: E402
from zvflare.psf_field import ZernikeField, make_zernike_field  # noqa: E402
from zvflare.sv_conv import sv_convolve_direct             # noqa: E402
from zvflare.utils import (                                # noqa: E402
    heatmap, hstack, load_image, make_light_sources, save_image,
    synth_night_background, to_uint8,
)


def uniform_field(image_hw, num_modes, coeffs) -> ZernikeField:
    """A constant (spatially-invariant) field == the Flare7K single-PSF model."""
    grid = np.tile(coeffs[None, None, :], (2, 2, 1)).astype(np.float32)
    return ZernikeField(coeffs=grid, image_hw=image_hw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--bg", default=None, help="optional background image")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    cfg = SynthConfig(image_size=args.size)
    synth = FlareSynthesizer(cfg)
    bg = load_image(args.bg, args.size) if args.bg else synth_night_background(args.size, rng)

    # ---- full end-to-end pair ----
    pair = synth.synthesize(np.random.default_rng(args.seed), background=bg)
    panel = hstack([
        pair["background"], pair["light_source"] / max(pair["light_source"].max(), 1e-6),
        pair["raw_flare"] / max(pair["raw_flare"].max(), 1e-6),
        pair["input"], pair["target"],
        np.repeat(pair["flare_mask"][..., None], 3, -1),
    ])
    save_image(os.path.join(args.out, "pair_panel.png"), panel)

    # ---- Fig.1-style comparison: uniform vs spatially-varying PSF ----
    size = args.size
    rng2 = np.random.default_rng(args.seed + 1)
    pts, glow, _ = make_light_sources(size, rng2, n=6)
    gen = synth.gen

    # single scalar PSF (mild fixed defocus) applied everywhere
    fixed = np.zeros(cfg.optics.num_zernike, np.float32)
    fixed[2] = 0.6  # defocus
    uni = uniform_field((size, size), cfg.optics.num_zernike, fixed)
    flare_uni = sv_convolve_direct(pts, uni, gen, tile=cfg.tile, overlap=cfg.overlap)

    # spatially-varying field
    sv = make_zernike_field((size, size), cfg.optics.num_zernike, grid=(7, 7),
                            strength=0.45, smooth=0.8, rng=rng2)
    flare_sv = sv_convolve_direct(pts, sv, gen, tile=cfg.tile, overlap=cfg.overlap)

    def lum(x):
        return x.mean(-1) if x.ndim == 3 else x

    cmp = hstack([
        flare_uni / max(flare_uni.max(), 1e-6),
        heatmap(lum(flare_uni)),
        flare_sv / max(flare_sv.max(), 1e-6),
        heatmap(lum(flare_sv)),
    ])
    save_image(os.path.join(args.out, "fig1_uniform_vs_sv.png"), cmp)

    # quantify asymmetry: radial-symmetry residual around each source
    def asymmetry(flare, positions):
        l = lum(flare)
        vals = []
        for (cy, cx) in positions:
            patch = l[max(0, cy - 30):cy + 30, max(0, cx - 30):cx + 30]
            if patch.size == 0:
                continue
            flipped = patch[::-1, ::-1]
            vals.append(np.abs(patch - flipped).sum() / (patch.sum() + 1e-8))
        return float(np.mean(vals)) if vals else 0.0

    _, _, pos = make_light_sources(size, np.random.default_rng(args.seed + 1), n=6)
    print("Point-symmetry residual (0 = perfectly centro-symmetric):")
    print("  uniform single PSF : %.4f" % asymmetry(flare_uni, pos))
    print("  spatially-varying  : %.4f" % asymmetry(flare_sv, pos))
    print("Saved:", os.path.join(args.out, "pair_panel.png"),
          "and", os.path.join(args.out, "fig1_uniform_vs_sv.png"))


if __name__ == "__main__":
    main()
