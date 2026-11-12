
from __future__ import annotations

import cv2
import numpy as np


def _validate_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3).")

    if not np.isfinite(image).all():
        raise ValueError("Image contains invalid numerical values.")

    if image.max() > 1.0:
        image = image / 255.0

    return np.clip(image, 0.0, 1.0)


def denoise(
    image: np.ndarray,
    strength: float = 0.5
) -> np.ndarray:
    """
    Conservative bilateral denoising for already-rendered RGB/sRGB.

    strength=0 returns the original image.
    strength=1 returns the full bilateral-filtered result.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1.")

    image_u8 = np.round(
        np.clip(image, 0.0, 1.0) * 255.0
    ).astype(np.uint8)

    filtered = cv2.bilateralFilter(
        image_u8,
        d=7,
        sigmaColor=35,
        sigmaSpace=35
    ).astype(np.float32) / 255.0

    result = (
        (1.0 - strength) * image
        + strength * filtered
    )

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def classical_isp(image: np.ndarray) -> np.ndarray:
    """
    Safe classical enhancement for an already-rendered RGB image.

    Does NOT apply RAW camera operations such as:
    white balance, colour correction matrix, gamma encoding,
    or tone mapping.
    """
    image = _validate_rgb(image)

    # Conservative edge-preserving denoising.
    output = denoise(image, strength=0.5)

    return np.clip(output, 0.0, 1.0).astype(np.float32)


def raw_to_rgb(
    raw: np.ndarray,
    bayer_pattern: str = "RGGB",
    black_level: float = 0.0,
    white_level: float = 1023.0
) -> np.ndarray:
    """
    Basic Bayer RAW normalization and demosaicing.

    This is a starting point, not a calibrated camera pipeline.
    """
    raw = np.asarray(raw)

    if raw.ndim != 2:
        raise ValueError("Expected a 2D Bayer RAW image.")

    if white_level <= black_level:
        raise ValueError("Invalid sensor white and black levels.")

    pattern_codes = {
        "RGGB": cv2.COLOR_BayerRG2RGB,
        "BGGR": cv2.COLOR_BayerBG2RGB,
        "GRBG": cv2.COLOR_BayerGR2RGB,
        "GBRG": cv2.COLOR_BayerGB2RGB
    }

    pattern = bayer_pattern.upper()

    if pattern not in pattern_codes:
        raise ValueError("Unsupported Bayer pattern.")

    normalized = (
        raw.astype(np.float32) - black_level
    ) / (white_level - black_level)

    normalized = np.clip(normalized, 0.0, 1.0)

    raw_u16 = np.round(
        normalized * 65535.0
    ).astype(np.uint16)

    rgb = cv2.cvtColor(
        raw_u16,
        pattern_codes[pattern]
    )

    return rgb.astype(np.float32) / 65535.0


def process_raw(
    raw: np.ndarray,
    bayer_pattern: str = "RGGB",
    black_level: float = 0.0,
    white_level: float = 1023.0
) -> np.ndarray:
    """Bayer RAW -> demosaicing -> basic RGB denoising."""
    rgb = raw_to_rgb(
        raw,
        bayer_pattern=bayer_pattern,
        black_level=black_level,
        white_level=white_level
    )

    return classical_isp(rgb)
