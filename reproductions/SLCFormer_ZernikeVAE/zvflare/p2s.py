"""Phase-to-Space (P2S) decomposition of spatially-varying PSFs.

SLCFormer Eq.(6) rewrites the per-pixel PSF as a shared set of spatial basis
functions with locally-varying coefficients:

    h_x(u) = sum_i beta_i(x) * phi_i(u)                         (6)

This is the Phase-to-Space transform of Mao/Chimitt/Chan (ICCV'21): instead of
rendering one Fourier-optics PSF per pixel, we
  1. build a dictionary of PSFs from random Zernike wavefronts,
  2. run PCA to obtain ``M`` spatial basis kernels ``phi_i`` (the "precomputed
     basis dictionary derived from empirical optical data"), and
  3. fit a mapping from Zernike coefficients ``alpha`` to PCA weights ``beta``.

The pay-off is the linearity used by :mod:`zvflare.sv_conv`:

    flare = sum_u h_x(u) L(x-u) = sum_i beta_i(x) * (phi_i (*) L)(x)

i.e. ``M`` global convolutions + a weighted sum, instead of a per-pixel kernel.

Following Mao et al., the global *tilt* of a PSF (which only translates it) is
split off as a centroid shift and decomposed separately from the *centered blur*
shape.  This keeps the linear PCA basis compact (a shift-variant signal is very
hard to span with few linear components).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import shift as nd_shift

from .optics import OpticsConfig, PSFGenerator

__all__ = ["P2SBasis", "build_p2s_basis", "psf_centroid"]


def psf_centroid(psf: np.ndarray) -> tuple[float, float]:
    """Intensity centroid (cy, cx) of a 2-D PSF."""
    ys, xs = np.mgrid[0:psf.shape[0], 0:psf.shape[1]]
    tot = psf.sum()
    if tot <= 0:
        c = (psf.shape[0] - 1) / 2.0
        return c, c
    return float((psf * ys).sum() / tot), float((psf * xs).sum() / tot)


@dataclass
class P2SBasis:
    """A fitted Phase-to-Space basis + Zernike->weight regressor.

    Attributes
    ----------
    mean : (k, k) mean PSF.
    components : (M, k, k) PCA spatial basis kernels phi_i.
    reg : (P, M) ridge-regression matrix mapping polynomial features of the
        Zernike coefficients to PCA weights beta (None if not fitted).
    poly_degree : degree of the polynomial feature expansion used by ``reg``.
    psf_size : kernel side length k.
    num_zernike : Zernike coefficient dimension expected by ``predict_weights``.
    """

    mean: np.ndarray
    components: np.ndarray
    reg: np.ndarray | None        # maps poly(alpha) -> beta
    shift_reg: np.ndarray | None  # maps poly(alpha) -> (dy, dx) centroid shift
    poly_degree: int
    psf_size: int
    num_zernike: int

    @property
    def center(self) -> float:
        return (self.psf_size - 1) / 2.0

    # ----------------------------------------------------------- projection
    def project(self, psf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decompose a PSF into (beta, shift).

        The PSF is first re-centred to its centroid; ``beta`` are the orthonormal
        PCA weights of the centred blur and ``shift=(dy,dx)`` is the centroid
        offset from the kernel centre.
        """
        cy, cx = psf_centroid(psf)
        dy, dx = cy - self.center, cx - self.center
        centred = nd_shift(psf, shift=(-dy, -dx), order=1, mode="constant")
        flat = (centred - self.mean).reshape(-1)
        comp = self.components.reshape(self.components.shape[0], -1)
        beta = comp @ flat  # orthonormal basis => projection is a dot product
        return beta, np.array([dy, dx], dtype=np.float32)

    def reconstruct(self, beta: np.ndarray, shift: np.ndarray | None = None) -> np.ndarray:
        comp = self.components.reshape(self.components.shape[0], -1)
        psf = (self.mean.reshape(-1) + beta @ comp).reshape(self.psf_size, self.psf_size)
        if shift is not None:
            psf = nd_shift(psf, shift=(float(shift[0]), float(shift[1])), order=1, mode="constant")
        return psf

    # --------------------------------------------------- learned alpha->beta
    def _poly_features(self, alpha: np.ndarray) -> np.ndarray:
        """[1, alpha, alpha^2] style features (degree up to ``poly_degree``)."""
        feats = [np.ones(alpha.shape[:-1] + (1,), dtype=alpha.dtype)]
        cur = alpha
        for _ in range(self.poly_degree):
            feats.append(cur)
            cur = cur * alpha  # elementwise power (keeps it cheap; no cross terms)
        return np.concatenate(feats, axis=-1)

    def predict_weights(self, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map Zernike coefficients ``alpha`` -> (beta, shift) (the P2S map)."""
        if self.reg is None:
            raise RuntimeError("P2S regressor not fitted; use project() instead.")
        feats = self._poly_features(np.asarray(alpha, dtype=np.float32))
        beta = feats @ self.reg
        shift = feats @ self.shift_reg if self.shift_reg is not None else None
        return beta, shift


def build_p2s_basis(
    gen: PSFGenerator,
    num_basis: int = 64,
    num_samples: int = 4000,
    strength: float = 0.5,
    poly_degree: int = 3,
    ridge: float = 1e-3,
    fit_regressor: bool = True,
    seed: int = 0,
) -> P2SBasis:
    """Sample a PSF dictionary, run PCA, and fit the Zernike->weight regressor.

    The generator is temporarily forced to monochromatic so the basis captures
    spatial structure; colour is reintroduced downstream.
    """
    rng = np.random.default_rng(seed)
    k = gen.cfg.psf_size

    # Force monochromatic rendering for the spatial basis.
    mono_cfg = OpticsConfig(**{**gen.cfg.__dict__})
    mono_cfg.chromatic = False
    mono = PSFGenerator(mono_cfg)

    center = (k - 1) / 2.0
    alphas = np.zeros((num_samples, gen.cfg.num_zernike), dtype=np.float32)
    psfs = np.zeros((num_samples, k * k), dtype=np.float32)
    shifts = np.zeros((num_samples, 2), dtype=np.float32)
    for s in range(num_samples):
        a = mono.sample_coeffs(rng, strength=strength)
        psf = mono.psf_from_coeffs(a)
        cy, cx = psf_centroid(psf)
        dy, dx = cy - center, cx - center
        # Re-centre the blur; the tilt-induced shift is modelled separately.
        psf_c = nd_shift(psf, shift=(-dy, -dx), order=1, mode="constant")
        alphas[s] = a
        psfs[s] = psf_c.reshape(-1)
        shifts[s] = (dy, dx)

    mean = psfs.mean(axis=0)
    X = psfs - mean
    # PCA via economy SVD; Vt rows are the principal spatial directions.
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    components = Vt[:num_basis]  # (M, k*k), orthonormal rows

    basis = P2SBasis(
        mean=mean.reshape(k, k),
        components=components.reshape(num_basis, k, k),
        reg=None,
        shift_reg=None,
        poly_degree=poly_degree,
        psf_size=k,
        num_zernike=gen.cfg.num_zernike,
    )
    if fit_regressor:
        beta = X @ components.T  # (num_samples, M) ground-truth weights
        F = basis._poly_features(alphas)  # (num_samples, P)
        A = F.T @ F + ridge * np.eye(F.shape[1], dtype=np.float32)
        basis.reg = np.linalg.solve(A, F.T @ beta).astype(np.float32)
        basis.shift_reg = np.linalg.solve(A, F.T @ shifts).astype(np.float32)
    return basis
