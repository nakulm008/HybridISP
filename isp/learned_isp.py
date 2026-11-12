
import numpy as np
import torch

from models.model_loader import model, DEVICE


def _tile_positions(length, tile_size):
    """
    Create evenly distributed tile positions.

    Avoids adding an unnecessary final tile immediately
    beside an already-covered tile.
    """
    if length <= tile_size:
        return [0]

    count = int(np.ceil(length / tile_size))

    positions = np.linspace(
        0,
        length - tile_size,
        count
    )

    return sorted(set(positions.round().astype(int).tolist()))


def learned_isp(
    image,
    task="sidd",
    tile_size=384,
    overlap=32,
    batch_size=1
):
    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with 3 channels.")

    if not np.isfinite(image).all():
        raise ValueError("Image contains invalid values.")

    image = np.clip(image, 0.0, 1.0)

    if not isinstance(tile_size, int) or tile_size <= 0:
        raise ValueError("tile_size must be positive.")

    if tile_size % 8 != 0:
        raise ValueError("tile_size must be divisible by 8.")

    if not isinstance(overlap, int) or overlap < 0 or overlap >= tile_size:
        raise ValueError("Invalid overlap.")

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be positive.")

    task_mapping = {
        "sidd": 0,
        "fivek": 1
    }

    if task not in task_mapping:
        raise ValueError("Task must be 'sidd' or 'fivek'.")

    task_value = task_mapping[task]

    height, width, _ = image.shape

    y_positions = _tile_positions(height, tile_size)
    x_positions = _tile_positions(width, tile_size)

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

            batch_array = np.stack(patches, axis=0)

            batch_array = np.ascontiguousarray(
                batch_array.transpose(0, 3, 1, 2)
            )

            tensor = torch.from_numpy(batch_array).to(
                DEVICE,
                memory_format=(
                    torch.channels_last
                    if DEVICE.type == "cpu"
                    else torch.contiguous_format
    )
)
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

            for index, (y, x) in enumerate(current_positions):

                patch_h, patch_w = patch_shapes[index]

                prediction = predictions[
                    index,
                    :patch_h,
                    :patch_w,
                    :
                ]

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

    return np.clip(result, 0.0, 1.0).astype(np.float32)
