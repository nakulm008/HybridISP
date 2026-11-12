
import cv2
import numpy as np

from isp.classical_isp import classical_isp
from isp.learned_isp import learned_isp


TILE_SIZE = 256
OVERLAP = 32
BATCH_SIZE = 1


def _validate_rgb(image):
    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape H x W x 3.")

    if not np.isfinite(image).all():
        raise ValueError("Image contains invalid numeric values.")

    if image.max() > 1.0:
        image = image / 255.0

    return np.clip(image, 0.0, 1.0)


def estimate_noise_map(image):
    """Estimate local high-frequency activity. This is a heuristic."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = np.abs(gray - smooth)

    noise_map = cv2.GaussianBlur(residual, (9, 9), 0)

    scale = np.percentile(noise_map, 95)

    if scale > 1e-6:
        noise_map = noise_map / scale

    return np.clip(noise_map, 0.0, 1.0)[..., None]

    scale = np.percentile(noise_map, 95)

    if scale > 1e-6:
        noise_map = noise_map / scale

    return np.clip(noise_map, 0.0, 1.0)[..., None]


def estimate_illumination_map(image):
    """Estimate local illumination using a smoothed grayscale image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    illumination = cv2.GaussianBlur(
        gray, (31, 31), 0
    )

    darkness = 1.0 - illumination

    return np.clip(darkness, 0.0, 1.0)[..., None]


def adaptive_fusion(
    image,
    learned_output,
    classical_output,
    task="sidd",
    base_learned_weight=0.85
):
    image = _validate_rgb(image)
    learned_output = _validate_rgb(learned_output)
    classical_output = _validate_rgb(classical_output)

    if learned_output.shape != image.shape:
        raise ValueError("Learned output dimensions do not match input.")

    if classical_output.shape != image.shape:
        raise ValueError("Classical output dimensions do not match input.")

    if task == "sidd":
        weight_map = (
            base_learned_weight
            + 0.10 * estimate_noise_map(image)
        )
    elif task == "fivek":
        weight_map = (
            base_learned_weight
            + 0.10 * estimate_illumination_map(image)
        )
    else:
        raise ValueError("Task must be 'sidd' or 'fivek'.")

    weight_map = np.clip(weight_map, 0.65, 0.95)
    weight_map = cv2.GaussianBlur(
        weight_map.astype(np.float32),
        (15, 15),
        0
    )

    if weight_map.ndim == 2:
        weight_map = weight_map[..., None]

    fused = (
        weight_map * learned_output
        + (1.0 - weight_map) * classical_output
    )

    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def fuse_outputs(
    image,
    learned_output,
    classical_output,
    task="sidd",
    learned_weight=0.85,
    adaptive=True
):
    """Fuse already-computed branch outputs without rerunning either model."""

    image = _validate_rgb(image)
    learned_output = _validate_rgb(learned_output)
    classical_output = _validate_rgb(classical_output)

    if learned_output.shape != image.shape:
        raise ValueError("Learned output dimensions do not match input.")

    if classical_output.shape != image.shape:
        raise ValueError("Classical output dimensions do not match input.")

    if adaptive:
        return adaptive_fusion(
            image,
            learned_output,
            classical_output,
            task=task,
            base_learned_weight=learned_weight
        )

    weight = float(np.clip(learned_weight, 0.0, 1.0))

    fused = (
        weight * learned_output
        + (1.0 - weight) * classical_output
    )

    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def hybrid_isp(
    image,
    task="sidd",
    learned_weight=0.85,
    adaptive=True
):
    """Standalone Hybrid ISP entry point."""

    image = _validate_rgb(image)

    classical_output = classical_isp(image)
    learned_output = learned_isp(image, task=task)

    return fuse_outputs(
        image,
        learned_output,
        classical_output,
        task=task,
        learned_weight=learned_weight,
        adaptive=adaptive
    )
