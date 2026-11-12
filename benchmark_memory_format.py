
import time
import statistics
import numpy as np
import torch

from models.model_loader import model, DEVICE

if DEVICE.type != "cpu":
    print("This benchmark is for CPU only.")
    raise SystemExit(0)

rng = np.random.default_rng(42)

# One representative CNN tile
array = rng.random(
    (1, 3, 256, 256),
    dtype=np.float32
)

print("Device:", DEVICE)
print("PyTorch:", torch.__version__)
print("MKLDNN enabled:", torch.backends.mkldnn.enabled)

for threads in [4, 8]:

    torch.set_num_threads(threads)

    for memory_format in [
        torch.contiguous_format,
        torch.channels_last
    ]:

        model.to(memory_format=memory_format)

        tensor = torch.from_numpy(array.copy())

        tensor = tensor.to(
            DEVICE,
            memory_format=memory_format
        )

        task_ids = torch.zeros(
            (1,),
            dtype=torch.long,
            device=DEVICE
        )

        # Warm-up
        with torch.inference_mode():
            for _ in range(2):
                output = model(tensor, task_ids)

        timings = []

        with torch.inference_mode():
            for _ in range(5):

                start = time.perf_counter()

                output = model(tensor, task_ids)

                elapsed = time.perf_counter() - start
                timings.append(elapsed)

        name = (
            "Channels Last"
            if memory_format == torch.channels_last
            else "Standard NCHW"
        )

        print(
            f"Threads: {threads} | "
            f"Format: {name} | "
            f"Average: {statistics.mean(timings):.4f}s | "
            f"Best: {min(timings):.4f}s"
        )

print("Memory-format benchmark completed.")
