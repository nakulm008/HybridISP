
import cv2
import numpy as np


def classical_isp(image):
    """
    Traditional edge-preserving image denoising.

    Input: RGB NumPy image, float32, range [0,1].
    Output: RGB NumPy image, float32, range [0,1].
    """

    image = np.asarray(image, dtype=np.float32)
    image = np.clip(image, 0, 1)

    image_uint8 = np.round(image * 255).astype(np.uint8)

    # Bilateral filtering preserves edges while reducing noise
    denoised = cv2.bilateralFilter(
        image_uint8,
        d=7,
        sigmaColor=35,
        sigmaSpace=35
    )

    result = denoised.astype(np.float32) / 255.0

    return np.clip(result, 0, 1)
