"""Flare7K++-style compositing of a synthetic scatter flare onto a background.

Training pairs follow the additive model used by Flare7K/Flare7K++ (Dai et al.),
which SLCFormer reuses: the operation is linear in the *de-gamma'd* domain, the
light source is preserved in the ground truth (so the network learns to keep
streetlights while removing their flare), and a battery of photometric/geometric
augmentations is applied to the flare layer.

    I_input  = gamma( clip( bg^g + F + S + noise ) ),  g ~ U(1.8, 2.2)
    I_target = gamma( clip( bg^g + S ) )               # flare removed, source kept

where ``F`` is the scatter-flare layer, ``S`` the light-source glow, and ``g`` the
random gamma.  Augmentation parameters mirror those listed in the paper
(inverse gamma, RGB scaling, colour jitter, geometric transforms, blur, noise).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter

__all__ = ["AugmentConfig", "linearize", "delinearize", "augment_flare", "compose_pair"]


@dataclass
class AugmentConfig:
    gamma_range: tuple = (1.8, 2.2)
    rgb_gain_range: tuple = (0.5, 1.2)      # per-channel flare gain (colour jitter)
    global_gain_range: tuple = (0.5, 1.5)   # overall flare brightness
    rotate_deg: float = 180.0               # +/- range
    scale_range: tuple = (0.7, 1.5)
    shear_deg: float = 10.0
    translate_frac: float = 0.15
    blur_sigma_range: tuple = (0.0, 2.0)
    noise_std: float = 0.01                 # gaussian read noise (linear domain)
    dc_offset_range: tuple = (0.0, 0.02)    # additive haze floor


def linearize(img: np.ndarray, gamma: float) -> np.ndarray:
    """sRGB-encoded [0,1] -> linear domain via I^gamma."""
    return np.clip(img, 0.0, 1.0) ** gamma


def delinearize(img: np.ndarray, gamma: float) -> np.ndarray:
    """Linear domain -> gamma-encoded [0,1]."""
    return np.clip(img, 0.0, None) ** (1.0 / gamma)


def _affine(img: np.ndarray, mat: np.ndarray, offset: np.ndarray) -> np.ndarray:
    out = np.empty_like(img)
    if img.ndim == 2:
        return affine_transform(img, mat, offset=offset, order=1, mode="constant")
    for c in range(img.shape[2]):
        out[..., c] = affine_transform(img[..., c], mat, offset=offset, order=1, mode="constant")
    return out


def augment_flare(
    flare: np.ndarray,
    cfg: AugmentConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply geometric + photometric augmentation to a (H,W,3) flare layer."""
    H, W = flare.shape[:2]
    f = flare.astype(np.float32)

    # --- geometric: rotation, scale, shear about the image centre ---
    ang = np.deg2rad(rng.uniform(-cfg.rotate_deg, cfg.rotate_deg))
    sc = rng.uniform(*cfg.scale_range)
    sh = np.deg2rad(rng.uniform(-cfg.shear_deg, cfg.shear_deg))
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
    Sh = np.array([[1.0, np.tan(sh)], [0.0, 1.0]], np.float32)
    M = (R @ Sh) / sc                       # inverse map (output->input) for affine_transform
    center = np.array([H / 2.0, W / 2.0])
    ty = rng.uniform(-cfg.translate_frac, cfg.translate_frac) * H
    tx = rng.uniform(-cfg.translate_frac, cfg.translate_frac) * W
    offset = center - M @ center - np.array([ty, tx])
    f = _affine(f, M, offset)

    # --- photometric: per-channel + global gain, DC haze ---
    gains = rng.uniform(*cfg.rgb_gain_range, size=3).astype(np.float32)
    gains *= rng.uniform(*cfg.global_gain_range)
    f = f * gains[None, None, :]
    f = f + rng.uniform(*cfg.dc_offset_range)

    # --- blur ---
    s = rng.uniform(*cfg.blur_sigma_range)
    if s > 0.05:
        f = gaussian_filter(f, sigma=(s, s, 0), mode="constant")
    return np.clip(f, 0.0, None)


def compose_pair(
    background: np.ndarray,
    flare: np.ndarray,
    light_source: np.ndarray | None,
    cfg: AugmentConfig,
    rng: np.random.Generator,
    augment: bool = True,
) -> dict:
    """Build a (input, target) training pair plus auxiliary layers.

    All inputs are gamma-encoded [0,1] float images of equal HxW; ``flare`` and
    ``light_source`` are linear-domain layers (as produced by the optics path).

    Returns a dict with keys: ``input``, ``target``, ``flare``, ``light_source``,
    ``gamma`` and ``flare_mask``.
    """
    H, W = background.shape[:2]
    g = float(rng.uniform(*cfg.gamma_range))
    bg_lin = linearize(background, g)

    F = augment_flare(flare, cfg, rng) if augment else flare.astype(np.float32)
    S = np.zeros_like(bg_lin) if light_source is None else light_source.astype(np.float32)

    noisy = rng.normal(0.0, cfg.noise_std, size=bg_lin.shape).astype(np.float32) if cfg.noise_std > 0 else 0.0

    input_lin = bg_lin + F + S + noisy
    target_lin = bg_lin + S                     # keep light source, drop flare

    out = {
        "input": np.clip(delinearize(input_lin, g), 0.0, 1.0),
        "target": np.clip(delinearize(target_lin, g), 0.0, 1.0),
        "flare": F,
        "light_source": S,
        "gamma": g,
        "flare_mask": (F.mean(axis=-1) > 0.02).astype(np.float32),
    }
    return out
