
import time
import gc
import statistics

import numpy as np
import torch

from models.model_loader import DEVICE
from isp.learned_isp import learned_isp


if DEVICE.type != "cpu":
    print("This benchmark is configured for CPU.")
    raise SystemExit(0)

torch.set_num_threads(8)

HEIGHT = 3000
WIDTH = 5328

print("=" * 55)
print("HYBRIDISP HIGH-RESOLUTION BENCHMARK")
print("=" * 55)
print("Resolution:", WIDTH, "x", HEIGHT)
print("Device:", DEVICE)
print("CPU threads:", torch.get_num_threads())

# Deterministic synthetic RGB image.
# This tests processing time, not image quality.
image = np.random.default_rng(42).random(
    (HEIGHT, WIDTH, 3),
    dtype=np.float32
)

results = []

for tile_size in [256, 384, 512]:

    print(f"\nTesting tile size: {tile_size} x {tile_size}")

    try:
        start = time.perf_counter()

        output = learned_isp(
            image,
            task="sidd",
            tile_size=tile_size,
            overlap=32,
            batch_size=1
        )

        elapsed = time.perf_counter() - start

        assert output.shape == image.shape

        results.append((tile_size, elapsed))

        print(f"Latency: {elapsed:.2f} seconds")
        print("Output shape:", output.shape)

        del output
        gc.collect()

    except Exception as error:
        print("FAILED:", str(error))
        gc.collect()

print("\n" + "=" * 55)
print("BENCHMARK SUMMARY")
print("=" * 55)

for tile_size, elapsed in results:
    print(
        f"Tile {tile_size:3d} | "
        f"Latency {elapsed:.2f} seconds"
    )

if results:
    best = min(results, key=lambda item: item[1])
    print("\nFastest tile size:", best[0])
    print("Best latency:", round(best[1], 2), "seconds")

print("=" * 55)
