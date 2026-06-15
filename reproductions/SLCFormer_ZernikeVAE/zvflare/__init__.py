"""zvflare -- a from-scratch reproduction of SLCFormer's ZernikeVAE-based
scatter-flare synthesis pipeline (AAAI 2026, arXiv:2512.15221).

Public API:
    PSFGenerator / OpticsConfig      -- Fourier-optics PSFs from Zernike phases
    make_zernike_field / ZernikeField -- spatially-varying phase field
    build_p2s_basis / P2SBasis        -- Phase-to-Space decomposition (Eq.6)
    sv_convolve_direct / sv_convolve_p2s -- spatially-varying convolution (Eq.3)
    FlareSynthesizer / SynthConfig    -- end-to-end synthesis + compositing
    ZernikeVAE / VAEConfig            -- stochastic flare refinement
"""

from .optics import OpticsConfig, PSFGenerator
from .psf_field import ZernikeField, make_zernike_field
from .p2s import P2SBasis, build_p2s_basis
from .sv_conv import sv_convolve_direct, sv_convolve_p2s
from .compositing import AugmentConfig, compose_pair
from .synth import FlareSynthesizer, SynthConfig
from .vae import VAEConfig, ZernikeVAE, vae_loss

__all__ = [
    "OpticsConfig", "PSFGenerator",
    "ZernikeField", "make_zernike_field",
    "P2SBasis", "build_p2s_basis",
    "sv_convolve_direct", "sv_convolve_p2s",
    "AugmentConfig", "compose_pair",
    "FlareSynthesizer", "SynthConfig",
    "VAEConfig", "ZernikeVAE", "vae_loss",
]

__version__ = "0.1.0"
