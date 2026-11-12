
import time
import statistics

import numpy as np
import torch

from models.model_loader import DEVICE
from isp.learned_isp import learned_isp


if DEVICE.type != "cpu":
    print("CPU benchmark only.")
    raise SystemExit(0)

torch.set_num_threads(8)

image = np.random.default_rng(42).random(
    (512, 512, 3),
    dtype=np.float32
)

print("Device:", DEVICE)
print("CPU threads:", torch.get_num_threads())
print("Image: 512 x 512")

for batch_size in [1, 2, 4, 8]:

    print(f"\nTesting batch size: {batch_size}")

    # Warm-up
    learned_isp(
        image,
        task="sidd",
        tile_size=256,
        batch_size=batch_size
    )

    timings = []

    for run in range(3):

        start = time.perf_counter()

        output = learned_isp(
            image,
            task="sidd",
            tile_size=256,
            batch_size=batch_size
        )

        elapsed = time.perf_counter() - start
        timings.append(elapsed)

    print(
        f"Average: {statistics.mean(timings):.3f} seconds | "
        f"Best: {min(timings):.3f} seconds"
    )

print("\nBatch benchmark completed.")
