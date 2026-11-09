
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from isp.classical_isp import classical_isp
from isp.learned_isp import learned_isp
from isp.hybrid_isp import hybrid_isp


# Replace with any RGB test image on your computer
IMAGE_PATH = "test_image.jpg"

image = np.asarray(
    Image.open(IMAGE_PATH).convert("RGB"),
    dtype=np.float32
) / 255.0

# Classical
start = time.perf_counter()
classical = classical_isp(image)
classical_time = time.perf_counter() - start

# Learned
start = time.perf_counter()
learned = learned_isp(image, task="sidd")
learned_time = time.perf_counter() - start

# Hybrid
start = time.perf_counter()
hybrid = hybrid_isp(image, task="sidd")
hybrid_time = time.perf_counter() - start

print("Classical:", classical.shape, f"{classical_time:.3f} seconds")
print("Learned:", learned.shape, f"{learned_time:.3f} seconds")
print("Hybrid:", hybrid.shape, f"{hybrid_time:.3f} seconds")

# Display comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, result, title in zip(
    axes.flat,
    [image, classical, learned, hybrid],
    ["Original", "Classical ISP", "Learned ISP", "Hybrid ISP"]
):
    ax.imshow(result)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()
