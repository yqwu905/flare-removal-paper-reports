"""End-to-end ZernikeVAE scatter-flare synthesis.

Ties together the four stages of SLCFormer's pipeline (Fig.2):

  1. Sample a spatially-varying Zernike phase field        (Eq.5, Fig.2 "Gaussian i.i.d.")
  2. Render PSFs by Fourier optics                          (Eq.1/2/4)
  3. Spatially-varying convolution of the light sources     (Eq.3, or Eq.6 via P2S)
  4. (optional) ZernikeVAE refinement                       (encoder-decoder + reparam.)

and finally composites a Flare7K++-style (input, target) training pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import zoom

from .compositing import AugmentConfig, compose_pair
from .optics import OpticsConfig, PSFGenerator
from .psf_field import make_zernike_field
from .sv_conv import sv_convolve_direct, sv_convolve_sources
from .utils import make_light_sources, synth_night_background

__all__ = ["SynthConfig", "FlareSynthesizer"]


@dataclass
class SynthConfig:
    image_size: int = 256
    optics: OpticsConfig = field(default_factory=lambda: OpticsConfig(
        pupil_size=96, psf_size=64, oversample=2, num_zernike=18, chromatic=True))
    field_grid: tuple = (6, 6)
    field_strength: float = 0.4
    field_smooth: float = 1.0
    backend: str = "sources"         # 'sources' (exact, sparse) or 'direct' (tiled)
    tile: int = 64
    overlap: float = 0.5
    n_light_sources: int = 5
    flare_gain: float = 6.0          # overall brightness of the scatter layer
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    # VAE descriptor settings
    descriptor_zernike_modes: int = 4  # dominant modes exposed to the VAE


class FlareSynthesizer:
    def __init__(self, cfg: SynthConfig | None = None):
        self.cfg = cfg or SynthConfig()
        self.gen = PSFGenerator(self.cfg.optics)

    # ----------------------------------------------------------- main entry
    def synthesize(
        self,
        rng: np.random.Generator,
        background: np.ndarray | None = None,
    ) -> dict:
        c = self.cfg
        size = c.image_size
        if background is None:
            background = synth_night_background(size, rng)

        # (1) light sources + (2/3) scatter flare via spatially-varying PSF
        pts, glow, positions = make_light_sources(size, rng, n=c.n_light_sources)
        zfield = make_zernike_field(
            (size, size), c.optics.num_zernike, grid=c.field_grid,
            strength=c.field_strength, smooth=c.field_smooth, rng=rng,
        )
        if c.backend == "sources":
            flare = sv_convolve_sources(pts, zfield, self.gen)
        else:
            flare = sv_convolve_direct(pts, zfield, self.gen, tile=c.tile, overlap=c.overlap)
        flare = self._norm_gain(flare) * c.flare_gain

        # auxiliary maps for the VAE descriptor
        kmap = self._kernel_size_map(zfield, size)
        zmaps = self._dominant_zernike_maps(zfield, size)

        # (4) composite the training pair
        pair = compose_pair(background, flare, glow, c.augment, rng, augment=True)
        pair.update({
            "background": background,
            "raw_flare": flare,
            "kernel_size_map": kmap,
            "zernike_maps": zmaps,
            "positions": positions,
            "zfield": zfield,
        })
        return pair

    # ------------------------------------------------- VAE descriptor stack
    def vae_descriptor(self, pair: dict, flare_gamma: float = 0.45) -> np.ndarray:
        """Stack [flare RGB, dominant Zernike maps, kernel-size map] -> (H,W,8).

        The flare RGB is tone-mapped with ``flare_gamma`` so the dim halo/streak
        structure (not just the saturated cores) is visible to the VAE -- without
        this the L1/L2 target is dominated by the dark background and the VAE
        collapses to black.
        """
        flare = pair["raw_flare"]
        flare = flare if flare.ndim == 3 else np.repeat(flare[..., None], 3, -1)
        flare = self._norm01(flare) ** flare_gamma
        z = pair["zernike_maps"]
        k = pair["kernel_size_map"][..., None]
        return np.concatenate([flare, z, k], axis=-1).astype(np.float32)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _norm_gain(x: np.ndarray) -> np.ndarray:
        m = x.max()
        return x / m if m > 0 else x

    @staticmethod
    def _norm01(x: np.ndarray) -> np.ndarray:
        lo, hi = float(x.min()), float(x.max())
        return (x - lo) / (hi - lo) if hi > lo else x

    def _kernel_size_map(self, zfield, size: int) -> np.ndarray:
        """Per-pixel local PSF spread (RMS radius), upsampled from the grid."""
        gh, gw, _ = zfield.coeffs.shape
        rad = np.zeros((gh, gw), np.float32)
        for i in range(gh):
            for j in range(gw):
                psf = self.gen.psf_from_coeffs(zfield.coeffs[i, j])
                lum = psf.mean(-1) if psf.ndim == 3 else psf
                rad[i, j] = self._rms_radius(lum)
        rad = zoom(rad, (size / gh, size / gw), order=1)
        return self._norm01(rad)

    def _dominant_zernike_maps(self, zfield, size: int) -> np.ndarray:
        """Upsample the first few Zernike coefficient maps to (H,W,M)."""
        m = self.cfg.descriptor_zernike_modes
        maps = zfield.coeffs[..., :m]
        gh, gw, _ = maps.shape
        dense = zoom(maps, (size / gh, size / gw, 1), order=1)
        # normalise each map to [0,1] for the network
        out = np.empty_like(dense)
        for k in range(dense.shape[-1]):
            out[..., k] = self._norm01(dense[..., k])
        return out.astype(np.float32)

    @staticmethod
    def _rms_radius(psf: np.ndarray) -> float:
        ys, xs = np.mgrid[0:psf.shape[0], 0:psf.shape[1]]
        tot = psf.sum()
        if tot <= 0:
            return 0.0
        cy, cx = (psf * ys).sum() / tot, (psf * xs).sum() / tot
        r2 = ((ys - cy) ** 2 + (xs - cx) ** 2) * psf
        return float(np.sqrt(r2.sum() / tot))
