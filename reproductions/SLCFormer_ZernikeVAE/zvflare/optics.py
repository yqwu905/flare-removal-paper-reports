"""Fourier-optics PSF generation from Zernike-parameterised wavefronts.

Implements SLCFormer Eq.(1), (2), (4):

    h(x)   = | F{ P(x) } |^2                                   (1)
    P(x)   = A(x) * exp(j * phi(x))                            (2)
    h_x(u) = | F{ A(x) * exp(-j 2pi phi_x(rho)) } |^2          (4)

``A`` is the (circular) aperture function, ``phi`` the pupil phase built from a
Zernike expansion (see :mod:`zvflare.zernike`), ``F`` the 2-D Fourier transform,
and ``h`` the intensity point-spread function (PSF).

The chromatic (RGB) extension scales the wavefront phase by ``lambda_ref/lambda``
per colour channel: a fixed physical optical-path-difference produces a larger
phase in waves at shorter wavelengths, which is the origin of the coloured
fringes seen in real scatter flares.  This is an optional, physically-motivated
addition on top of the (monochromatic) model written in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .zernike import zernike_basis

__all__ = ["OpticsConfig", "PSFGenerator"]

# Reference RGB wavelengths in nanometres (sRGB-ish primaries).
RGB_WAVELENGTHS_NM = (610.0, 540.0, 465.0)


@dataclass
class OpticsConfig:
    """Configuration for the Fourier-optics PSF generator."""

    pupil_size: int = 256          # sampling of the pupil plane (square grid)
    psf_size: int = 64             # cropped PSF kernel size returned to the caller
    oversample: int = 2            # zero-pad factor for the pupil FFT (PSF sampling)
    num_zernike: int = 28          # number of Zernike modes (Noll j = 2 .. 2+N-1)
    phase_scale: float = 2.0 * np.pi  # maps coefficient units -> radians (Eq.4 uses 2pi)
    aperture: str = "circular"     # 'circular' or 'gaussian-apodized'
    apod_sigma: float = 0.7        # apodisation sigma (in pupil radii) if gaussian
    chromatic: bool = True         # produce a 3-channel (RGB) chromatic PSF
    wavelengths_nm: tuple = field(default_factory=lambda: RGB_WAVELENGTHS_NM)
    ref_wavelength_nm: float = 540.0


class PSFGenerator:
    """Build PSFs from Zernike coefficient vectors.

    The Zernike basis and aperture are precomputed once; each call to
    :meth:`psf_from_coeffs` is a couple of FFTs.
    """

    def __init__(self, cfg: OpticsConfig | None = None):
        self.cfg = cfg or OpticsConfig()
        c = self.cfg
        self.basis, self.mask = zernike_basis(c.num_zernike, c.pupil_size, start=2)
        self.aperture = self._build_aperture()

    # ------------------------------------------------------------------ build
    def _build_aperture(self) -> np.ndarray:
        c = self.cfg
        size = c.pupil_size
        ax = np.linspace(-1.0, 1.0, size)
        xx, yy = np.meshgrid(ax, ax)
        r = np.sqrt(xx * xx + yy * yy)
        A = (r <= 1.0).astype(np.float32)
        if c.aperture == "gaussian-apodized":
            A = A * np.exp(-(r ** 2) / (2.0 * c.apod_sigma ** 2)).astype(np.float32)
        return A

    # ------------------------------------------------------------------- core
    def phase_from_coeffs(self, coeffs: np.ndarray) -> np.ndarray:
        """phi(rho) = sum_i a_i Z_i(rho)  (Eq.5), returned on the pupil grid."""
        coeffs = np.asarray(coeffs, dtype=np.float32)
        n = min(len(coeffs), self.basis.shape[0])
        # einsum over the used modes -> (pupil_size, pupil_size)
        return np.einsum("i,ijk->jk", coeffs[:n], self.basis[:n])

    def _psf_mono(self, phase: np.ndarray, wave_scale: float = 1.0) -> np.ndarray:
        """Monochromatic PSF: h = |F{A exp(j * phase * wave_scale)}|^2  (Eq.1/4)."""
        c = self.cfg
        P = self.aperture * np.exp(1j * c.phase_scale * wave_scale * phase)
        # Zero-pad the pupil so the PSF is sampled finely enough.
        pad = c.pupil_size * (c.oversample - 1) // 2
        Pp = np.pad(P, pad, mode="constant")
        field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Pp)))
        psf = np.abs(field) ** 2
        psf = self._center_crop(psf, c.psf_size)
        s = psf.sum()
        if s > 0:
            psf = psf / s  # energy normalisation (sum to 1)
        return psf.astype(np.float32)

    def psf_from_coeffs(self, coeffs: np.ndarray) -> np.ndarray:
        """Return the PSF for a Zernike coefficient vector.

        Returns a ``(psf_size, psf_size)`` array (monochromatic) or, when
        ``cfg.chromatic`` is set, a ``(psf_size, psf_size, 3)`` RGB PSF.
        """
        phase = self.phase_from_coeffs(coeffs)
        if not self.cfg.chromatic:
            return self._psf_mono(phase)
        chans = []
        for lam in self.cfg.wavelengths_nm:
            wave_scale = self.cfg.ref_wavelength_nm / lam
            chans.append(self._psf_mono(phase, wave_scale))
        psf = np.stack(chans, axis=-1)
        # Re-normalise so the luminance sums to 1 (keeps compositing energy stable).
        s = psf.sum(axis=(0, 1), keepdims=True)
        s[s == 0] = 1.0
        return (psf / s).astype(np.float32)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _center_crop(a: np.ndarray, size: int) -> np.ndarray:
        h, w = a.shape[:2]
        top = (h - size) // 2
        left = (w - size) // 2
        return a[top:top + size, left:left + size]

    # Convenience: sample random aberration coefficients (Gaussian i.i.d., Fig.2).
    def sample_coeffs(
        self,
        rng: np.random.Generator,
        strength: float = 0.4,
        decay: float = 0.7,
    ) -> np.ndarray:
        """Draw Zernike coefficients ~ N(0, sigma_i^2).

        ``sigma_i`` decays with radial order (``decay`` per order) so low-order
        aberrations (tilt/defocus/astigmatism) dominate, matching real lens
        wavefronts and the turbulence prior used by the Phase-to-Space transform.
        """
        from .zernike import noll_to_nm

        n = self.cfg.num_zernike
        sig = np.empty(n, dtype=np.float32)
        for k in range(n):
            order, _ = noll_to_nm(2 + k)
            sig[k] = strength * (decay ** (order - 1))
        return (rng.standard_normal(n).astype(np.float32) * sig)
