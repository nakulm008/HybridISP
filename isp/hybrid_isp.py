
import numpy as np

from isp.classical_isp import classical_isp
from isp.learned_isp import learned_isp


# Shared CPU inference configuration
TILE_SIZE = 256
OVERLAP = 32
BATCH_SIZE = 1


def hybrid_isp(
    image,
    task="sidd",
    learned_weight=0.85
):
    """
    Hybrid ISP pipeline.

    Combines:
        85% learned ISP output
        15% classical ISP output

    Uses the same optimized tile configuration
    as the standalone learned ISP evaluation.
    """

    if not 0.0 <= learned_weight <= 1.0:
        raise ValueError(
            "learned_weight must be between 0 and 1."
        )

    image = np.asarray(
        image,
        dtype=np.float32
    )

    image = np.clip(image, 0.0, 1.0)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected an RGB image with 3 channels."
        )

    # Learned ISP with explicit optimized configuration.
    learned_output = learned_isp(
        image,
        task=task,
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        batch_size=BATCH_SIZE
    )

    # Classical ISP.
    classical_output = classical_isp(image)

    if learned_output.shape != classical_output.shape:
        raise ValueError(
            "Learned and classical outputs have different shapes."
        )

    classical_weight = 1.0 - learned_weight

    # Weighted fusion.
    hybrid_output = (
        learned_weight * learned_output
        + classical_weight * classical_output
    )

    return np.clip(
        hybrid_output,
        0.0,
        1.0
    )
