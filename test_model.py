import torch
from models.model_loader import model, DEVICE

test_image = torch.rand(
    1, 3, 128, 128,
    device=DEVICE
)

task_id = torch.tensor(
    [0],
    dtype=torch.long,
    device=DEVICE
)

with torch.inference_mode():
    output = model(test_image, task_id)

print("Input:", test_image.shape)
print("Output:", output.shape)
print("Device:", DEVICE)
print("Model integration successful.")