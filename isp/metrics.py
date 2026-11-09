
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def calculate_metrics(prediction, reference):
    """
    Calculate PSNR and SSIM between an ISP output
    and its ground-truth reference image.

    Both images must be RGB float arrays in [0, 1]
    with identical dimensions.
    """

    prediction = np.asarray(prediction, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)

    if prediction.shape != reference.shape:
        raise ValueError(
            f"Image dimensions do not match: "
            f"prediction={prediction.shape}, "
            f"reference={reference.shape}"
        )

    if prediction.ndim != 3 or prediction.shape[2] != 3:
        raise ValueError("Images must be RGB arrays with 3 channels.")

    prediction = np.clip(prediction, 0.0, 1.0)
    reference = np.clip(reference, 0.0, 1.0)

    psnr = peak_signal_noise_ratio(
        reference,
        prediction,
        data_range=1.0
    )

    ssim = structural_similarity(
        reference,
        prediction,
        channel_axis=-1,
        data_range=1.0
    )

    return {
        "psnr": round(float(psnr), 4),
        "ssim": round(float(ssim), 4)
    }


def evaluate_all_outputs(outputs, reference):
    """
    Evaluate multiple ISP outputs against one reference.
    """

    results = {}

    for name, image in outputs.items():
        results[name] = calculate_metrics(image, reference)

    return results

import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def calculate_metrics(prediction, reference):
    """
    Calculate PSNR and SSIM between an ISP output
    and its ground-truth reference image.

    Both images must be RGB float arrays in [0, 1]
    with identical dimensions.
    """

    prediction = np.asarray(prediction, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)

    if prediction.shape != reference.shape:
        raise ValueError(
            f"Image dimensions do not match: "
            f"prediction={prediction.shape}, "
            f"reference={reference.shape}"
        )

    if prediction.ndim != 3 or prediction.shape[2] != 3:
        raise ValueError("Images must be RGB arrays with 3 channels.")

    prediction = np.clip(prediction, 0.0, 1.0)
    reference = np.clip(reference, 0.0, 1.0)

    psnr = peak_signal_noise_ratio(
        reference,
        prediction,
        data_range=1.0
    )

    ssim = structural_similarity(
        reference,
        prediction,
        channel_axis=-1,
        data_range=1.0
    )

    return {
        "psnr": round(float(psnr), 4),
        "ssim": round(float(ssim), 4)
    }


def evaluate_all_outputs(outputs, reference):
    """
    Evaluate multiple ISP outputs against one reference.
    """

    results = {}

    for name, image in outputs.items():
        results[name] = calculate_metrics(image, reference)

    return results
