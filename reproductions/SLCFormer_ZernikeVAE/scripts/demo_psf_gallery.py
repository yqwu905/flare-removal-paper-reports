"""Demo: visualise individual Zernike aberration modes and the PSFs they produce.

Produces two panels in ``assets/``:
  * ``zernike_modes.png`` -- the low-order Zernike phase maps (tilt, defocus, ...).
  * ``psf_gallery.png``   -- chromatic PSFs for a sweep of single-mode aberrations
                             plus a few random multi-mode wavefronts.

Usage:
    python scripts/demo_psf_gallery.py --out assets
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zvflare.optics import OpticsConfig, PSFGenerator        # noqa: E402
from zvflare.zernike import ANSI_NAMES, zernike_basis         # noqa: E402
from zvflare.utils import hstack, save_image                  # noqa: E402


def colorize_phase(phase, mask):
    import matplotlib
    p = phase.copy()
    lo, hi = p[mask].min(), p[mask].max()
    p = (p - lo) / max(hi - lo, 1e-8)
    rgb = matplotlib.colormaps["twilight"](p)[..., :3]
    rgb[~mask] = 1.0
    return rgb.astype(np.float32)


def norm_psf(psf):
    psf = psf / max(psf.max(), 1e-8)
    return psf ** 0.45  # gamma for display


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- Zernike phase maps ----
    size = 128
    basis, mask = zernike_basis(num_modes=10, size=size, start=2)
    mode_imgs = [colorize_phase(basis[i], mask) for i in range(8)]
    save_image(os.path.join(args.out, "zernike_modes.png"), hstack(mode_imgs))

    # ---- PSF gallery ----
    gen = PSFGenerator(OpticsConfig(pupil_size=128, psf_size=96, oversample=3,
                                    num_zernike=20, chromatic=True))
    row = []
    # single-mode sweeps (defocus j=4 -> idx2, coma j=7 -> idx5, astig j=5 -> idx3)
    specs = [("airy", {}), ("defocus", {2: 1.2}), ("astig", {3: 1.4}),
             ("coma", {5: 1.6}), ("trefoil", {7: 1.8}), ("spherical", {9: 1.5})]
    for _name, mods in specs:
        c = np.zeros(gen.cfg.num_zernike, np.float32)
        for k, v in mods.items():
            c[k] = v
        row.append(norm_psf(gen.psf_from_coeffs(c)))
    # random multi-mode wavefronts
    rng = np.random.default_rng(0)
    for _ in range(4):
        row.append(norm_psf(gen.psf_from_coeffs(gen.sample_coeffs(rng, strength=0.5))))
    save_image(os.path.join(args.out, "psf_gallery.png"), hstack(row))

    print("Zernike modes shown:", [ANSI_NAMES.get(2 + i, "?") for i in range(8)])
    print("Saved zernike_modes.png and psf_gallery.png to", args.out)


if __name__ == "__main__":
    main()
