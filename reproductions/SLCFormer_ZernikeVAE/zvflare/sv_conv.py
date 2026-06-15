"""Spatially-varying convolution: turn a light-source map into a scatter flare.

SLCFormer Eq.(3) models the corrupted image as a position-dependent convolution
of the clean signal with a per-pixel PSF:

    I(x_i) = sum_j  h_{x_i}(u_j) * I_clear(u_j)                 (3)

For flare synthesis the "clean signal" that scatters is the *light-source map*
``L`` (bright streetlights/headlights on a dark night).  Convolving ``L`` with
the spatially-varying PSF field gives the scatter flare layer.

Three backends are provided:

* ``sv_convolve_sources`` -- **exact & default** for sparse light sources.  Each
  source is spread by the PSF rendered at its own location; no tiling error.
* ``sv_convolve_direct`` -- tiled reference for continuous light maps.  Each tile
  is FFT-convolved with the PSF at its centre and Hann-blended (overlap-add).
  Accurate only when the tile is >= the PSF spread.
* ``sv_convolve_p2s`` -- the accelerator of Eq.(6).  Using ``h_x = sum_i
  beta_i(x) phi_i``, the SV convolution becomes ``flare = sum_i beta_i ⊙
  (phi_i (*) L)`` -- ``M`` global FFT convolutions plus a weighted sum, a cost
  independent of how finely the PSF varies (approximate; linear PCA basis).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom
from scipy.signal import fftconvolve

from .optics import PSFGenerator
from .p2s import P2SBasis
from .psf_field import ZernikeField

__all__ = ["sv_convolve_direct", "sv_convolve_sources", "sv_convolve_p2s"]


def _hann2d(h: int, w: int) -> np.ndarray:
    wy = np.hanning(h + 2)[1:-1]
    wx = np.hanning(w + 2)[1:-1]
    win = np.outer(wy, wx).astype(np.float32)
    win[win < 1e-6] = 1e-6
    return win


def _as3(x: np.ndarray) -> np.ndarray:
    return x[..., None] if x.ndim == 2 else x


def sv_convolve_direct(
    light: np.ndarray,
    field: ZernikeField,
    gen: PSFGenerator,
    tile: int = 128,
    overlap: float = 0.5,
) -> np.ndarray:
    """Exact tile-wise spatially-varying convolution (Eq.3/4).

    Parameters
    ----------
    light : (H, W) or (H, W, 3) light-source intensity map (linear domain).
    field : spatially-varying Zernike coefficient field.
    gen : PSF generator (its ``chromatic`` flag decides the output channels).
    tile : tile side length in pixels.
    overlap : fractional overlap between neighbouring tiles (Hann blended).
    """
    light = _as3(light).astype(np.float32)
    H, W, C = light.shape
    step = max(1, int(tile * (1.0 - overlap)))
    out_C = 3 if gen.cfg.chromatic else 1   # output channels follow the PSF
    out = np.zeros((H, W, out_C), dtype=np.float32)
    wsum = np.zeros((H, W, 1), dtype=np.float32)
    win = _hann2d(tile, tile)[..., None]

    ys = list(range(0, max(1, H - tile + 1), step))
    xs = list(range(0, max(1, W - tile + 1), step))
    if ys[-1] != H - tile:
        ys.append(max(0, H - tile))
    if xs[-1] != W - tile:
        xs.append(max(0, W - tile))

    for ty in ys:
        for tx in xs:
            cy, cx = ty + tile / 2.0, tx + tile / 2.0
            coeffs = field.sample(cy, cx)
            psf = gen.psf_from_coeffs(coeffs)          # (k,k) or (k,k,3)
            psf = _as3(psf)
            patch = light[ty:ty + tile, tx:tx + tile]
            ph, pw = patch.shape[:2]
            acc = np.zeros((ph, pw, out.shape[2]), dtype=np.float32)
            for c in range(out.shape[2]):
                lc = patch[..., min(c, C - 1)]
                pc = psf[..., min(c, psf.shape[2] - 1)]
                acc[..., c] = fftconvolve(lc, pc, mode="same")
            w = win[:ph, :pw]
            out[ty:ty + tile, tx:tx + tile] += acc * w
            wsum[ty:ty + tile, tx:tx + tile] += w
    out /= np.maximum(wsum, 1e-6)
    return out[..., 0] if out.shape[2] == 1 else out


def sv_convolve_sources(
    points: np.ndarray,
    field: ZernikeField,
    gen: PSFGenerator,
    threshold: float = 1e-4,
) -> np.ndarray:
    """Exact source-centric SV convolution for a *sparse* light-source map.

    Each non-zero source at ``p`` is spread by the PSF rendered at its own
    location ``h_p = h(field.sample(p))`` and accumulated centred at ``p``.  This
    is an exact realisation of Eq.(3) for point sources (no tiling approximation),
    cheap when light sources are sparse (streetlights/headlights), and it is the
    default path used by :class:`~zvflare.synth.FlareSynthesizer`.
    """
    points = points if points.ndim == 2 else points.mean(axis=-1)
    H, W = points.shape
    out_C = 3 if gen.cfg.chromatic else 1
    out = np.zeros((H, W, out_C), dtype=np.float32)
    k = gen.cfg.psf_size
    half = k // 2

    ys, xs = np.where(points > threshold)
    for y, x in zip(ys, xs):
        inten = float(points[y, x])
        psf = gen.psf_from_coeffs(field.sample(y, x))
        psf = psf[..., None] if psf.ndim == 2 else psf
        # destination window (clamped to image) and matching PSF crop
        y0, y1 = y - half, y - half + k
        x0, x1 = x - half, x - half + k
        dy0, dx0 = max(0, y0), max(0, x0)
        dy1, dx1 = min(H, y1), min(W, x1)
        sy0, sx0 = dy0 - y0, dx0 - x0
        sy1, sx1 = sy0 + (dy1 - dy0), sx0 + (dx1 - dx0)
        if dy1 <= dy0 or dx1 <= dx0:
            continue
        out[dy0:dy1, dx0:dx1, :] += inten * psf[sy0:sy1, sx0:sx1, :]
    return out[..., 0] if out_C == 1 else out


def sv_convolve_p2s(
    light: np.ndarray,
    field: ZernikeField,
    basis: P2SBasis,
    gen: PSFGenerator | None = None,
    weights: str = "project",
) -> np.ndarray:
    """Accelerated SV convolution via the Phase-to-Space basis (Eq.6).

    Uses the linearity ``flare = mean(*)L + sum_i beta_i ⊙ (phi_i (*) L)`` so the
    cost is ``M+1`` global FFT convolutions regardless of how finely the PSF
    varies -- the regime where this beats the tile-wise direct path is *dense*
    (near per-pixel) PSF variation.

    The per-pixel weights ``beta_i(x)`` are obtained at the (coarse) control-grid
    nodes and bilinearly interpolated:

    * ``weights="project"`` (default, accurate): render the node PSF by Fourier
      optics and project it onto the basis (needs ``gen``).
    * ``weights="learned"``: use the fitted alpha->beta regressor (no FFT).

    NB: this linear form models the *centred-blur* component; the per-pixel tilt
    (a translation of the PSF) is handled separately as a displacement field,
    following Mao et al.  Zero the tilt modes of ``field`` for an apples-to-apples
    comparison with :func:`sv_convolve_direct`.
    """
    light = light if light.ndim == 2 else light.mean(axis=-1)
    light = light.astype(np.float32)
    H, W = light.shape
    gh, gw, _ = field.coeffs.shape
    M = basis.components.shape[0]

    # weights at grid nodes -> (gh, gw, M)
    beta_grid = np.zeros((gh, gw, M), np.float32)
    if weights == "project":
        if gen is None:
            raise ValueError("weights='project' requires a PSFGenerator")
        mono = gen
        for i in range(gh):
            for j in range(gw):
                psf = mono.psf_from_coeffs(field.coeffs[i, j])
                psf = psf.mean(-1) if psf.ndim == 3 else psf
                b, _sh = basis.project(psf)
                beta_grid[i, j] = b
    elif weights == "learned":
        b, _sh = basis.predict_weights(field.coeffs.reshape(-1, field.coeffs.shape[-1]))
        beta_grid = b.reshape(gh, gw, M)
    else:
        raise ValueError("weights must be 'project' or 'learned'")

    # bilinearly upsample weight maps to per-pixel
    beta = zoom(beta_grid, (H / gh, W / gw, 1), order=1)

    flare = fftconvolve(light, basis.mean, mode="same").astype(np.float32)  # mean term
    for i in range(M):
        conv_i = fftconvolve(light, basis.components[i], mode="same")
        flare += beta[..., i] * conv_i
    return flare
