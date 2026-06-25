"""Compatibility imports for the renamed Gonzalez denoising module.

New code should import from detector.core.gonzalez_denoising. The old module
path remains only so older sessions and scripts do not fail at import time.
"""

from .gonzalez_denoising import (  # noqa: F401
    denoise_gonzalez_adaptive_wavelet,
    denoise_trace_gonzalez_full_trace,
    normalize_gonzalez_denoising_method,
)
