
import os
from pathlib import Path

import torch

from models.hybrid_isp_model import HybridISPNet


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "best_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DEFAULT_CPU_THREADS = min(8, os.cpu_count() or 1)

try:
    CPU_THREADS = int(
        os.environ.get(
            "HYBRIDISP_CPU_THREADS",
            DEFAULT_CPU_THREADS
        )
    )
except ValueError:
    CPU_THREADS = DEFAULT_CPU_THREADS

CPU_THREADS = max(1, CPU_THREADS)

if DEVICE.type == "cpu":
    torch.set_num_threads(CPU_THREADS)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def load_isp_model():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    config = checkpoint.get("model_config", {})

    isp_model = HybridISPNet(
        base_channels=config.get("base_channels", 32),
        num_tasks=config.get("num_tasks", 2)
    )

    isp_model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    isp_model.to(DEVICE)

    if DEVICE.type == "cpu":
        isp_model.to(
            memory_format=torch.channels_last
        )

    isp_model.eval()

    for parameter in isp_model.parameters():
        parameter.requires_grad_(False)

    return isp_model


model = load_isp_model()

MODEL_PARAMETERS = sum(
    p.numel() for p in model.parameters()
)

MODEL_SIZE_MB = (
    CHECKPOINT_PATH.stat().st_size / (1024 * 1024)
)

print("=" * 45)
print("HYBRIDISP MODEL INITIALIZED")
print("=" * 45)
print("Device:", DEVICE)
print("CPU threads:", torch.get_num_threads())
print("Parameters:", MODEL_PARAMETERS)
print("Checkpoint size:", round(MODEL_SIZE_MB, 2), "MB")
print("Model mode: Evaluation")
print("Gradients: Disabled")
print("=" * 45)
