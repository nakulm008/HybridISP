
import numpy as np

from isp.metrics import calculate_metrics

# Simulated reference image
reference = np.random.rand(128, 128, 3).astype(np.float32)

# Perfect prediction
perfect = reference.copy()

# Slightly corrupted prediction
noisy = np.clip(
    reference + np.random.normal(
        0, 0.03, reference.shape
    ).astype(np.float32),
    0,
    1
)

print("Perfect image:")
print(calculate_metrics(perfect, reference))

print("\nNoisy image:")
print(calculate_metrics(noisy, reference))
