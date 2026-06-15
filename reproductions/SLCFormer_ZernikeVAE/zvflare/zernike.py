"""Zernike polynomial basis on the unit disk.

This implements the orthonormal Zernike polynomials used to parameterise the
per-pixel phase function in SLCFormer Eq.(5):

    phi_x(rho) = sum_i a_i * Z_i(rho)

where ``Z_i`` are orthogonal Zernike polynomials representing aberration modes
(tilt, defocus, astigmatism, coma, and higher-order distortions) and ``a_i`` are
the coefficients.  We use Noll single-indexing (j = 1, 2, 3, ...) and Noll's
normalisation so the polynomials are orthonormal over the unit disk, which makes
the coefficients directly interpretable in RMS-wavefront units.

References
----------
* R. J. Noll, "Zernike polynomials and atmospheric turbulence", JOSA 1976.
* Mao, Chimitt, Chan, "Accelerating Atmospheric Turbulence Simulation via
  Learned Phase-to-Space Transform", ICCV 2021 (the Zernike -> PSF basis used by
  SLCFormer's synthesis pipeline).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

__all__ = [
    "noll_to_nm",
    "zernike_radial",
    "zernike_single",
    "zernike_basis",
    "ANSI_NAMES",
]

# Human-readable names for the first low-order Noll modes (for plots / debugging).
ANSI_NAMES = {
    1: "piston",
    2: "tilt-x",
    3: "tilt-y",
    4: "defocus",
    5: "astig-oblique",
    6: "astig-vertical",
    7: "coma-y",
    8: "coma-x",
    9: "trefoil-y",
    10: "trefoil-x",
    11: "spherical",
    12: "2nd-astig-v",
    13: "2nd-astig-o",
    14: "quadrafoil-x",
    15: "quadrafoil-y",
}


@lru_cache(maxsize=4096)
def noll_to_nm(j: int) -> tuple[int, int]:
    """Convert a Noll single index ``j`` (1-based) to ``(n, m)``.

    ``n`` is the radial order, ``m`` the (signed) azimuthal frequency.  The sign
    of ``m`` encodes cosine (m > 0) vs sine (m < 0) terms following Noll's
    convention.
    """
    if j < 1:
        raise ValueError("Noll index j must be >= 1")
    n = 0
    j1 = j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m = (-1) ** j * ((n % 2) + 2 * int((j1 + ((n + 1) % 2)) / 2))
    return n, m


def _norm(n: int, m: int) -> float:
    """Noll normalisation factor making the basis orthonormal on the unit disk."""
    if m == 0:
        return math.sqrt(n + 1)
    return math.sqrt(2 * (n + 1))


def zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Radial polynomial ``R_n^|m|(rho)`` evaluated on ``rho`` (array)."""
    m = abs(m)
    if (n - m) % 2 != 0:
        return np.zeros_like(rho)
    R = np.zeros_like(rho)
    half = (n - m) // 2
    for k in range(half + 1):
        c = (
            (-1) ** k
            * math.factorial(n - k)
            / (
                math.factorial(k)
                * math.factorial((n + m) // 2 - k)
                * math.factorial((n - m) // 2 - k)
            )
        )
        R += c * rho ** (n - 2 * k)
    return R


def zernike_single(j: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Evaluate the j-th (Noll) orthonormal Zernike polynomial.

    Values outside the unit disk (``rho > 1``) are returned as 0.
    """
    n, m = noll_to_nm(j)
    R = zernike_radial(n, m, rho)
    if m == 0:
        Z = R
    elif m > 0:
        Z = R * np.cos(m * theta)
    else:
        Z = R * np.sin(-m * theta)
    Z = _norm(n, m) * Z
    Z = np.where(rho <= 1.0, Z, 0.0)
    return Z


def _disk_grid(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rho, theta, mask) sampled on [-1, 1]^2 with a unit-disk mask."""
    ax = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(ax, ax)
    rho = np.sqrt(xx * xx + yy * yy)
    theta = np.arctan2(yy, xx)
    mask = rho <= 1.0
    return rho, theta, mask


def zernike_basis(num_modes: int, size: int, start: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Stack the first ``num_modes`` Zernike modes on a ``size x size`` grid.

    Parameters
    ----------
    num_modes : number of modes to return.
    size : spatial resolution of the (square) pupil sampling grid.
    start : first Noll index to include.  Defaults to 2 to skip piston (j=1),
        which only adds a constant phase offset and does not affect the PSF.

    Returns
    -------
    basis : (num_modes, size, size) float32 array, zeroed outside the unit disk.
    mask : (size, size) bool array, the unit-disk support.
    """
    rho, theta, mask = _disk_grid(size)
    basis = np.zeros((num_modes, size, size), dtype=np.float32)
    for k in range(num_modes):
        j = start + k
        basis[k] = zernike_single(j, rho, theta).astype(np.float32)
    return basis, mask
