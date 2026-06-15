"""Spatially-varying field of Zernike coefficients.

SLCFormer revisits the degradation with a *set of spatially varying PSFs*
(Eq.3): every output pixel ``x`` owns a phase ``phi_x`` and therefore a distinct
PSF ``h_x``.  Rendering a PSF per pixel is wasteful, so we store the Zernike
coefficients on a coarse control grid and bilinearly interpolate them.  The grid
is generated with spatial smoothness so neighbouring PSFs vary gradually --- this
is what produces the *non-uniform, non-centrally-symmetric* glow the paper argues
real flares exhibit (and Flare7K's single scalar PSF cannot).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

__all__ = ["ZernikeField", "make_zernike_field"]


@dataclass
class ZernikeField:
    """A (gh, gw, K) grid of Zernike coefficient vectors over the image."""

    coeffs: np.ndarray            # (gh, gw, K)
    image_hw: tuple[int, int]     # (H, W) the field maps onto

    @property
    def num_modes(self) -> int:
        return self.coeffs.shape[-1]

    def sample(self, y: float, x: float) -> np.ndarray:
        """Bilinearly sample the coefficient vector at image coordinate (y, x)."""
        gh, gw, _ = self.coeffs.shape
        H, W = self.image_hw
        # Map image coords -> grid coords.
        gy = y / max(H - 1, 1) * (gh - 1)
        gx = x / max(W - 1, 1) * (gw - 1)
        y0, x0 = int(np.floor(gy)), int(np.floor(gx))
        y1, x1 = min(y0 + 1, gh - 1), min(x0 + 1, gw - 1)
        wy, wx = gy - y0, gx - x0
        c00, c01 = self.coeffs[y0, x0], self.coeffs[y0, x1]
        c10, c11 = self.coeffs[y1, x0], self.coeffs[y1, x1]
        top = c00 * (1 - wx) + c01 * wx
        bot = c10 * (1 - wx) + c11 * wx
        return top * (1 - wy) + bot * wy

    def dense(self) -> np.ndarray:
        """Upsample the coefficient grid to per-pixel maps (H, W, K)."""
        gh, gw, K = self.coeffs.shape
        H, W = self.image_hw
        return zoom(self.coeffs, (H / gh, W / gw, 1), order=1)


def make_zernike_field(
    image_hw: tuple[int, int],
    num_modes: int,
    grid: tuple[int, int] = (8, 8),
    strength: float = 0.35,
    decay: float = 0.7,
    smooth: float = 1.0,
    rng: np.random.Generator | None = None,
) -> ZernikeField:
    """Generate a smooth, spatially-varying Zernike coefficient field.

    Coefficients are drawn i.i.d. Gaussian per grid node (Fig.2 "Gaussian
    i.i.d."), scaled per mode by ``strength * decay**order`` so low-order
    aberrations dominate, then spatially smoothed so adjacent PSFs are coherent.
    """
    rng = rng or np.random.default_rng()
    gh, gw = grid
    from .zernike import noll_to_nm

    sig = np.array(
        [strength * (decay ** (noll_to_nm(2 + k)[0] - 1)) for k in range(num_modes)],
        dtype=np.float32,
    )
    coeffs = rng.standard_normal((gh, gw, num_modes)).astype(np.float32) * sig
    if smooth > 0:
        # Smooth spatially (not across the mode axis) for coherent local PSFs.
        coeffs = gaussian_filter(coeffs, sigma=(smooth, smooth, 0), mode="nearest")
    return ZernikeField(coeffs=coeffs, image_hw=image_hw)
