"""Build and cache the Phase-to-Space basis (Eq.6) and report its fidelity.

Saves ``assets/p2s_basis.npz`` and a visualisation of the leading spatial basis
kernels.  Also prints the energy captured vs. number of basis kernels, and the
speed-up of the P2S SV-convolution over the direct tile-wise reference.

Usage:
    python scripts/build_p2s_basis.py --num-basis 64 --num-samples 3000
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zvflare.optics import OpticsConfig, PSFGenerator           # noqa: E402
from zvflare.p2s import build_p2s_basis                          # noqa: E402
from zvflare.psf_field import make_zernike_field                 # noqa: E402
from zvflare.sv_conv import sv_convolve_sources, sv_convolve_p2s  # noqa: E402
from zvflare.utils import hstack, make_light_sources, save_image, heatmap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets")
    ap.add_argument("--num-basis", type=int, default=64)
    ap.add_argument("--num-samples", type=int, default=3000)
    ap.add_argument("--strength", type=float, default=0.3)
    ap.add_argument("--num-zernike", type=int, default=18)
    ap.add_argument("--psf-size", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gen = PSFGenerator(OpticsConfig(pupil_size=96, psf_size=args.psf_size,
                                    oversample=2, num_zernike=args.num_zernike,
                                    chromatic=False))
    t = time.time()
    basis = build_p2s_basis(gen, num_basis=args.num_basis, num_samples=args.num_samples,
                            strength=args.strength, poly_degree=3, seed=0)
    print("Built P2S basis (%d kernels from %d samples) in %.1fs"
          % (args.num_basis, args.num_samples, time.time() - t))

    # fidelity on held-out PSFs
    rng = np.random.default_rng(123)
    ep, epr = [], []
    for _ in range(300):
        a = gen.sample_coeffs(rng, strength=args.strength)
        psf = gen.psf_from_coeffs(a)
        b, sh = basis.project(psf)
        bp, shp = basis.predict_weights(a)
        e = (psf ** 2).sum()
        ep.append(((basis.reconstruct(b, sh) - psf) ** 2).sum() / e)
        epr.append(((basis.reconstruct(bp, shp) - psf) ** 2).sum() / e)
    print("  energy captured  (PCA+shift projection):     %.1f%%" % (100 * (1 - np.mean(ep))))
    print("  energy captured  (learned alpha->beta map):  %.1f%%" % (100 * (1 - np.mean(epr))))

    np.savez(os.path.join(args.out, "p2s_basis.npz"),
             mean=basis.mean, components=basis.components,
             reg=basis.reg, shift_reg=basis.shift_reg,
             poly_degree=basis.poly_degree, psf_size=basis.psf_size,
             num_zernike=basis.num_zernike)

    # visualise leading basis kernels
    comps = [heatmap(basis.components[i]) for i in range(min(8, args.num_basis))]
    save_image(os.path.join(args.out, "p2s_basis_kernels.png"), hstack(comps))

    # agreement vs the EXACT source-centric SV-conv (ground truth).
    # Tilt modes are zeroed so the linear P2S form (centred blur) is comparable;
    # per-pixel tilt is a separate displacement field (see sv_convolve_p2s docs).
    size = 256
    pts, _, _ = make_light_sources(size, np.random.default_rng(5), n=8)
    field = make_zernike_field((size, size), args.num_zernike, grid=(8, 8),
                               strength=args.strength, smooth=1.2,
                               rng=np.random.default_rng(5))
    field.coeffs[..., 0:2] = 0.0   # zero tilt-x, tilt-y
    t0 = time.time(); f_exact = sv_convolve_sources(pts, field, gen); te = time.time() - t0
    f_exact = f_exact.mean(-1) if f_exact.ndim == 3 else f_exact
    t0 = time.time(); f_p2s = sv_convolve_p2s(pts, field, basis, gen=gen, weights="project"); tp = time.time() - t0
    rel = np.abs(f_p2s - f_exact).sum() / (np.abs(f_exact).sum() + 1e-8)
    print("Exact source SV-conv: %.3fs | P2S SV-conv: %.3fs | rel-L1 diff %.3f (M=%d FFTs, granularity-independent)"
          % (te, tp, rel, args.num_basis))
    f_direct = f_exact
    cmp = hstack([heatmap(f_direct), heatmap(f_p2s)])
    save_image(os.path.join(args.out, "p2s_vs_direct.png"), cmp)
    print("Saved p2s_basis.npz, p2s_basis_kernels.png, p2s_vs_direct.png to", args.out)


if __name__ == "__main__":
    main()
