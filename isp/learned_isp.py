
import numpy as np
import torch

from models.model_loader import model, DEVICE


def learned_isp(
    image,
    task="sidd",
    tile_size=256,
    overlap=32,
    batch_size=2
):
    """
    Apply HybridISPNet using batched overlapping tile inference.

    Tasks:
        sidd  -> task ID 0
        fivek -> task ID 1

    Defaults:
        tile_size = 256
        overlap = 32
        batch_size = 2
    """

    image = np.asarray(image, dtype=np.float32)
    image = np.clip(image, 0.0, 1.0)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with 3 channels.")

    if not isinstance(tile_size, int) or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer.")

    if tile_size % 8 != 0:
        raise ValueError("tile_size must be divisible by 8.")

    if not isinstance(overlap, int) or overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be >= 0 and < tile_size.")

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    task_mapping = {
        "sidd": 0,
        "fivek": 1
    }

    if task not in task_mapping:
        raise ValueError("Task must be 'sidd' or 'fivek'.")

    task_value = task_mapping[task]

    height, width, _ = image.shape
    stride = tile_size - overlap

    # Calculate tile positions and ensure complete image coverage.
    y_positions = list(
        range(0, max(height - tile_size, 0) + 1, stride)
    )
    x_positions = list(
        range(0, max(width - tile_size, 0) + 1, stride)
    )

    final_y = max(height - tile_size, 0)
    final_x = max(width - tile_size, 0)

    if not y_positions or y_positions[-1] != final_y:
        y_positions.append(final_y)

    if not x_positions or x_positions[-1] != final_x:
        x_positions.append(final_x)

    positions = [
        (y, x)
        for y in y_positions
        for x in x_positions
    ]

    output_sum = np.zeros(
        (height, width, 3),
        dtype=np.float32
    )

    weight_sum = np.zeros(
        (height, width, 1),
        dtype=np.float32
    )

    model.eval()

    with torch.inference_mode():

        # Process multiple tiles in each model call.
        for start in range(0, len(positions), batch_size):

            current_positions = positions[
                start:start + batch_size
            ]

            patches = []
            patch_shapes = []

            for y, x in current_positions:

                patch = image[
                    y:min(y + tile_size, height),
                    x:min(x + tile_size, width)
                ]

                patch_h, patch_w = patch.shape[:2]
                patch_shapes.append((patch_h, patch_w))

                # Pad edge tiles to the required tile dimensions.
                pad_h = tile_size - patch_h
                pad_w = tile_size - patch_w

                if pad_h or pad_w:
                    patch = np.pad(
                        patch,
                        (
                            (0, pad_h),
                            (0, pad_w),
                            (0, 0)
                        ),
                        mode="edge"
                    )

                patches.append(patch)

            # NHWC -> NCHW.
            batch_array = np.stack(patches, axis=0)
            batch_array = np.ascontiguousarray(
                batch_array.transpose(0, 3, 1, 2)
            )

            tensor = torch.from_numpy(batch_array).to(DEVICE)

            # Supply one task ID per image in the batch.
            task_ids = torch.full(
                (len(current_positions),),
                task_value,
                dtype=torch.long,
                device=DEVICE
            )

            predictions = model(tensor, task_ids)

            predictions = (
                predictions
                .float()
                .cpu()
                .numpy()
                .transpose(0, 2, 3, 1)
            )

            # Blend each predicted tile into the output.
            for index, (y, x) in enumerate(current_positions):

                patch_h, patch_w = patch_shapes[index]

                prediction = predictions[
                    index,
                    :patch_h,
                    :patch_w,
                    :
                ]

                # Non-zero Hann weights prevent uncovered borders.
                wy = np.hanning(patch_h + 2)[1:-1]
                wx = np.hanning(patch_w + 2)[1:-1]

                weights = np.outer(wy, wx)[..., None].astype(
                    np.float32
                )

                y2 = y + patch_h
                x2 = x + patch_w

                output_sum[y:y2, x:x2] += prediction * weights
                weight_sum[y:y2, x:x2] += weights

    result = output_sum / np.maximum(weight_sum, 1e-8)

    return np.clip(result, 0.0, 1.0)
