"""Batch dataset generator: produce many (input, target) flare-removal pairs.

The physics + flares are fully synthetic (no Flare7K templates needed), so the
ONLY thing you may want to supply is a folder of clean background photos.  With
no ``--backgrounds`` the script falls back to a procedural night background and
still runs end-to-end.

Output layout (Flare7K-style):
    <out>/input/00000.png        # flare-corrupted (network input)
    <out>/target/00000.png       # flare-free GT (light source preserved)
    <out>/flare/00000.png        # (optional) the scatter-flare layer
    <out>/mask/00000.png         # (optional) flare region mask
    <out>/meta.csv               # per-sample gamma / background file / seed

Usage:
    # zero-asset smoke run
    python scripts/generate_dataset.py --num 20 --out data/demo

    # realistic data from your own backgrounds (e.g. Flickr-24K)
    python scripts/generate_dataset.py --backgrounds /path/to/flickr24k \\
        --num 5000 --size 512 --out data/train --save-flare --save-mask
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zvflare import FlareSynthesizer, SynthConfig          # noqa: E402
from zvflare.optics import OpticsConfig                     # noqa: E402
from zvflare.utils import load_image, save_image            # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def list_backgrounds(folder: str | None) -> list[str]:
    if not folder:
        return []
    files = [f for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
             if f.lower().endswith(IMG_EXT)]
    return sorted(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--num", type=int, default=100, help="number of pairs")
    ap.add_argument("--size", type=int, default=512, help="output resolution")
    ap.add_argument("--backgrounds", default=None,
                    help="folder of clean background images (else synthetic)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["sources", "direct"], default="sources")
    ap.add_argument("--n-sources", type=int, default=5, help="light sources / image")
    ap.add_argument("--flare-gain", type=float, default=6.0)
    ap.add_argument("--save-flare", action="store_true", help="also save flare layer")
    ap.add_argument("--save-mask", action="store_true", help="also save flare mask")
    ap.add_argument("--psf-size", type=int, default=64)
    args = ap.parse_args()

    bgs = list_backgrounds(args.backgrounds)
    if args.backgrounds and not bgs:
        sys.exit(f"No images found under {args.backgrounds!r} ({', '.join(IMG_EXT)})")
    print(f"Backgrounds: {'synthetic (no folder given)' if not bgs else f'{len(bgs)} files from '+args.backgrounds}")

    cfg = SynthConfig(
        image_size=args.size,
        backend=args.backend,
        n_light_sources=args.n_sources,
        flare_gain=args.flare_gain,
        optics=OpticsConfig(pupil_size=96, psf_size=args.psf_size, oversample=2,
                            num_zernike=18, chromatic=True),
    )
    synth = FlareSynthesizer(cfg)

    for sub in ("input", "target") + (("flare",) if args.save_flare else ()) + (("mask",) if args.save_mask else ()):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    meta_path = os.path.join(args.out, "meta.csv")
    with open(meta_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["index", "background", "gamma", "seed"])
        for i in tqdm(range(args.num), desc="generating"):
            seed = args.seed * 1_000_003 + i
            rng = np.random.default_rng(seed)
            if bgs:
                bg_path = bgs[rng.integers(len(bgs))]
                bg = load_image(bg_path, args.size)
            else:
                bg_path, bg = "synthetic", None
            pair = synth.synthesize(rng, background=bg)

            name = f"{i:05d}.png"
            save_image(os.path.join(args.out, "input", name), pair["input"])
            save_image(os.path.join(args.out, "target", name), pair["target"])
            if args.save_flare:
                fl = pair["raw_flare"]
                save_image(os.path.join(args.out, "flare", name), fl / max(fl.max(), 1e-6))
            if args.save_mask:
                save_image(os.path.join(args.out, "mask", name), pair["flare_mask"])
            writer.writerow([i, os.path.basename(str(bg_path)), f"{pair['gamma']:.3f}", seed])

    print(f"Wrote {args.num} pairs to {args.out}/  (input/, target/"
          + (", flare/" if args.save_flare else "")
          + (", mask/" if args.save_mask else "") + ", meta.csv)")


if __name__ == "__main__":
    main()
