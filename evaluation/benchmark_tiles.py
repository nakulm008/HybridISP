
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Add project root to Python import path.
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from isp.learned_isp import learned_isp


DATA_DIR = PROJECT_DIR / "data" / "SIDD_Small_sRGB_Only" / "Data"

image_path = next(DATA_DIR.rglob("NOISY_SRGB_*.PNG"))

image = np.asarray(
    Image.open(image_path).convert("RGB"),
    dtype=np.float32
) / 255.0

print("Image:", image_path.name)
print("Shape:", image.shape)
print("Device: CPU/GPU as configured in model_loader")
print()

configs = [
    (128, 32, 1),
    (256, 32, 1),
    (256, 32, 2),
    (512, 32, 1),
]

for tile_size, overlap, batch_size in configs:

    print(
        f"Testing tile={tile_size}, "
        f"overlap={overlap}, "
        f"batch={batch_size}",
        flush=True
    )

    start = time.perf_counter()

    output = learned_isp(
        image,
        task="sidd",
        tile_size=tile_size,
        overlap=overlap,
        batch_size=batch_size
    )

    elapsed = time.perf_counter() - start

    print(f"Latency: {elapsed:.2f} seconds")
    print("Output shape:", output.shape)
    print("-" * 50, flush=True)
