"""I/O, colour and visualisation helpers."""

from __future__ import annotations

import numpy as np

__all__ = [
    "load_image", "save_image", "to_uint8", "synth_night_background",
    "make_light_sources", "heatmap", "hstack",
]


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def load_image(path: str, size: int | None = None) -> np.ndarray:
    """Load an image as float32 RGB in [0,1]."""
    from PIL import Image

    im = Image.open(path).convert("RGB")
    if size is not None:
        im = im.resize((size, size), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def save_image(path: str, img: np.ndarray) -> None:
    from PIL import Image

    arr = to_uint8(img)
    if arr.ndim == 2:
        Image.fromarray(arr, mode="L").save(path)
    else:
        Image.fromarray(arr, mode="RGB").save(path)


def synth_night_background(
    size: int,
    rng: np.random.Generator,
    n_blobs: int = 40,
) -> np.ndarray:
    """A cheap synthetic 'night city' background (dark with dim coloured blobs).

    Lets the pipeline run end-to-end with zero external assets; replace with real
    Flickr-24K crops for serious training.
    """
    img = np.zeros((size, size, 3), np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    base = 0.02 + 0.03 * rng.random()
    img += base
    for _ in range(n_blobs):
        cy, cx = rng.uniform(0, size, 2)
        s = rng.uniform(4, 30)
        col = rng.uniform(0.05, 0.4, 3) * np.array([1.0, 0.9, 0.7])
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
        img += g[..., None] * col[None, None, :]
    # a faint horizon gradient
    img += np.linspace(0.0, 0.06, size)[:, None, None]
    return np.clip(img, 0.0, 1.0)


def make_light_sources(
    size: int,
    rng: np.random.Generator,
    n: int = 5,
    core_sigma: float = 1.2,
    glow_sigma: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create (point_map, source_glow, positions).

    ``point_map`` is the delta-like input to the scatter convolution; ``source_glow``
    is the bright (clipped) light-source layer kept in the GT.
    """
    from scipy.ndimage import gaussian_filter

    pts = np.zeros((size, size), np.float32)
    glow = np.zeros((size, size, 3), np.float32)
    positions = []
    for _ in range(n):
        cy, cx = int(rng.uniform(size * 0.1, size * 0.9)), int(rng.uniform(size * 0.1, size * 0.9))
        inten = rng.uniform(0.6, 1.0)
        pts[cy, cx] += inten
        positions.append((cy, cx))
        col = np.array([1.0, rng.uniform(0.8, 1.0), rng.uniform(0.6, 0.95)], np.float32)
        core = np.zeros((size, size), np.float32)
        core[cy, cx] = inten
        core = gaussian_filter(core, core_sigma)
        core = core / max(core.max(), 1e-8) * inten
        glow += core[..., None] * col[None, None, :]
    return pts, np.clip(glow, 0.0, 4.0), np.array(positions)


def heatmap(gray: np.ndarray, cmap: str = "magma") -> np.ndarray:
    """Map a 2-D array to an RGB heatmap (matplotlib colormap, no figure)."""
    import matplotlib

    g = gray.astype(np.float32)
    g = (g - g.min()) / max(g.max() - g.min(), 1e-8)
    rgb = matplotlib.colormaps[cmap](g)[..., :3]
    return rgb.astype(np.float32)


def hstack(images: list[np.ndarray], pad: int = 4, bg: float = 1.0) -> np.ndarray:
    """Horizontally concatenate equal-height RGB images with padding."""
    imgs = [im if im.ndim == 3 else np.repeat(im[..., None], 3, axis=-1) for im in images]
    h = max(im.shape[0] for im in imgs)
    out = []
    for im in imgs:
        if im.shape[0] != h:
            pad_h = h - im.shape[0]
            im = np.pad(im, ((0, pad_h), (0, 0), (0, 0)), constant_values=bg)
        out.append(im)
        out.append(np.full((h, pad, 3), bg, np.float32))
    return np.concatenate(out[:-1], axis=1)
