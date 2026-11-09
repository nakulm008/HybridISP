
from pathlib import Path
import torch

from models.hybrid_isp_model import HybridISPNet


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "best_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_isp_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    config = checkpoint.get("model_config", {})

    model = HybridISPNet(
        base_channels=32,
        num_tasks=2
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


model = load_isp_model()

print("HybridISP model loaded.")
print("Device:", DEVICE)
print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)
