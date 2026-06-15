"""Sanity tests for the zvflare synthesis pipeline.

Run with:  python -m pytest tests/ -q   (or)   python tests/test_pipeline.py
These check the *physics* invariants and the plumbing, not exact numbers.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zvflare.optics import OpticsConfig, PSFGenerator
from zvflare.zernike import zernike_basis, noll_to_nm
from zvflare.psf_field import make_zernike_field
from zvflare.sv_conv import sv_convolve_sources
from zvflare import FlareSynthesizer, SynthConfig


def test_noll_indexing():
    assert noll_to_nm(1) == (0, 0)
    assert noll_to_nm(2) == (1, 1)
    assert noll_to_nm(4) == (2, 0)      # defocus
    assert noll_to_nm(11) == (4, 0)     # spherical


def test_zernike_orthonormal():
    B, _ = zernike_basis(num_modes=10, size=256, start=1)
    da = (2.0 / 256) ** 2
    G = np.einsum("ijk,ljk->il", B, B) * da / np.pi
    assert np.allclose(np.diag(G), 1.0, atol=0.03)
    off = G - np.diag(np.diag(G))
    assert np.abs(off).max() < 0.03


def test_psf_energy_and_airy():
    g = PSFGenerator(OpticsConfig(pupil_size=128, psf_size=64, oversample=3, chromatic=False))
    psf = g.psf_from_coeffs(np.zeros(g.cfg.num_zernike, np.float32))
    assert abs(psf.sum() - 1.0) < 1e-3                 # energy conservation
    cy, cx = np.unravel_index(np.argmax(psf), psf.shape)
    assert abs(cy - 32) <= 1 and abs(cx - 32) <= 1     # Airy peak centred


def test_defocus_broadens():
    g = PSFGenerator(OpticsConfig(pupil_size=128, psf_size=64, oversample=3, chromatic=False))

    def rms(p):
        ys, xs = np.mgrid[0:p.shape[0], 0:p.shape[1]]
        cy, cx = (p * ys).sum(), (p * xs).sum()
        return np.sqrt((((ys - cy) ** 2 + (xs - cx) ** 2) * p).sum())

    z = np.zeros(g.cfg.num_zernike, np.float32)
    r0 = rms(g.psf_from_coeffs(z))
    z[2] = 1.5
    assert rms(g.psf_from_coeffs(z)) > 2.0 * r0


def test_chromatic_channels():
    g = PSFGenerator(OpticsConfig(pupil_size=96, psf_size=48, oversample=2, chromatic=True))
    psf = g.psf_from_coeffs(g.sample_coeffs(np.random.default_rng(0), 0.5))
    assert psf.shape == (48, 48, 3)
    assert np.allclose(psf.sum(axis=(0, 1)), 1.0, atol=1e-3)


def test_spatial_variation():
    """Two well-separated locations must yield different PSFs."""
    g = PSFGenerator(OpticsConfig(pupil_size=96, psf_size=48, num_zernike=18, chromatic=False))
    f = make_zernike_field((256, 256), 18, grid=(6, 6), strength=0.5,
                           rng=np.random.default_rng(1))
    p1 = g.psf_from_coeffs(f.sample(30, 30))
    p2 = g.psf_from_coeffs(f.sample(220, 220))
    tv = np.abs(p1 - p2).sum() / 2.0
    assert tv > 0.1


def test_synth_pair_shapes_and_range():
    synth = FlareSynthesizer(SynthConfig(image_size=128, n_light_sources=4))
    pair = synth.synthesize(np.random.default_rng(0))
    for k in ("input", "target"):
        assert pair[k].shape == (128, 128, 3)
        assert pair[k].min() >= 0.0 and pair[k].max() <= 1.0 + 1e-5
    assert synth.vae_descriptor(pair).shape[-1] == 8


def test_flare_increases_brightness():
    """The flared input should be at least as bright as the clean target."""
    synth = FlareSynthesizer(SynthConfig(image_size=128, n_light_sources=5))
    pair = synth.synthesize(np.random.default_rng(2))
    assert pair["input"].mean() >= pair["target"].mean() - 1e-3


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
            passed += 1
        except AssertionError as e:
            print("FAIL", fn.__name__, "->", e)
        except Exception as e:  # noqa: BLE001
            print("ERROR", fn.__name__, "->", type(e).__name__, e)
    print(f"\n{passed}/{len(fns)} tests passed")
    sys.exit(0 if passed == len(fns) else 1)
