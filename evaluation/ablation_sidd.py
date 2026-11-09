
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from models.model_loader import DEVICE
from isp.classical_isp import classical_isp
from isp.learned_isp import learned_isp
from isp.metrics import calculate_metrics


# Configuration
DATASET_DIR = (
    PROJECT_DIR / "data" / "SIDD_Small_sRGB_Only" / "Data"
)

OUTPUT_DIR = PROJECT_DIR / "evaluation" / "results"

CSV_PATH = OUTPUT_DIR / "sidd_ablation.csv"
JSON_PATH = OUTPUT_DIR / "sidd_ablation_summary.json"

TASK = "sidd"

# Learned ISP contribution in each experiment
FUSION_WEIGHTS = {
    "classical_only": None,
    "learned_only": None,
    "hybrid_85_15": 0.85,
    "hybrid_70_30": 0.70,
    "hybrid_50_50": 0.50,
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rgb(path):
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def sync_device():
    if str(DEVICE).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def discover_pairs():

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_DIR}"
        )

    pairs = []

    for scene in sorted(DATASET_DIR.iterdir()):

        if not scene.is_dir():
            continue

        for noisy in sorted(scene.glob("NOISY_SRGB_*.PNG")):

            suffix = noisy.name.replace("NOISY_SRGB_", "")
            reference = scene / f"GT_SRGB_{suffix}"

            if reference.exists():
                pairs.append((noisy, reference))

    return pairs


def measure(function):

    sync_device()
    start = time.perf_counter()

    result = function()

    sync_device()
    elapsed = (time.perf_counter() - start) * 1000

    return result, elapsed


def evaluate():

    pairs = discover_pairs()

    if not pairs:
        raise RuntimeError("No valid SIDD pairs found.")

    print("=" * 70)
    print("HYBRIDISP — ABLATION STUDY")
    print("=" * 70)
    print("Valid pairs:", len(pairs))
    print("Device:", DEVICE)

    # Warmup
    warmup = load_rgb(pairs[0][0])
    learned_isp(warmup, task=TASK)
    sync_device()

    records = []

    for index, (noisy_path, gt_path) in enumerate(pairs, 1):

        print(f"\n[{index}/{len(pairs)}] {noisy_path.name}")

        noisy = load_rgb(noisy_path)
        reference = load_rgb(gt_path)

        if noisy.shape != reference.shape:
            print("Skipped: dimension mismatch")
            continue

        # Run the two base pipelines once.
        classical, classical_time = measure(
            lambda: classical_isp(noisy)
        )

        learned, learned_time = measure(
            lambda: learned_isp(noisy, task=TASK)
        )

        configurations = {
            "classical_only": classical,
            "learned_only": learned,
            "hybrid_85_15": np.clip(
                0.85 * learned + 0.15 * classical,
                0, 1
            ),
            "hybrid_70_30": np.clip(
                0.70 * learned + 0.30 * classical,
                0, 1
            ),
            "hybrid_50_50": np.clip(
                0.50 * learned + 0.50 * classical,
                0, 1
            ),
        }

        # Fusion is a combination of already calculated outputs.
        # Its latency is therefore measured separately.
        fusion_start = time.perf_counter()

        fusion_outputs = {
            "hybrid_85_15": np.clip(
                0.85 * learned + 0.15 * classical, 0, 1
            ),
            "hybrid_70_30": np.clip(
                0.70 * learned + 0.30 * classical, 0, 1
            ),
            "hybrid_50_50": np.clip(
                0.50 * learned + 0.50 * classical, 0, 1
            ),
        }

        fusion_time = (
            time.perf_counter() - fusion_start
        ) * 1000

        # Use end-to-end pipeline latency for each configuration.
        latencies = {
            "classical_only": classical_time,
            "learned_only": learned_time,
            "hybrid_85_15": classical_time + learned_time,
            "hybrid_70_30": classical_time + learned_time,
            "hybrid_50_50": classical_time + learned_time,
        }

        for name, output in configurations.items():

            quality = calculate_metrics(output, reference)

            records.append({
                "scene": noisy_path.parent.name,
                "image": noisy_path.name,
                "experiment": name,
                "psnr_db": quality["psnr"],
                "ssim": quality["ssim"],
                "latency_ms": round(latencies[name], 3),
                "fusion_calculation_ms": round(
                    fusion_time if name.startswith("hybrid") else 0,
                    3
                )
            })

        print(f"Classical latency: {classical_time:.2f} ms")
        print(f"Learned latency: {learned_time:.2f} ms")

    if not records:
        raise RuntimeError("No images evaluated successfully.")

    # CSV
    fields = [
        "scene",
        "image",
        "experiment",
        "psnr_db",
        "ssim",
        "latency_ms",
        "fusion_calculation_ms"
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    # Aggregate
    summary = {}

    for experiment in FUSION_WEIGHTS:

        rows = [
            row for row in records
            if row["experiment"] == experiment
        ]

        summary[experiment] = {
            "image_count": len(rows),
            "average_psnr_db": round(
                statistics.mean(r["psnr_db"] for r in rows), 4
            ),
            "average_ssim": round(
                statistics.mean(r["ssim"] for r in rows), 4
            ),
            "average_latency_ms": round(
                statistics.mean(r["latency_ms"] for r in rows), 3
            )
        }

    with open(JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 85)
    print("ABLATION RESULTS")
    print("=" * 85)

    print(
        f"{'Experiment':<20}"
        f"{'PSNR (dB)':>15}"
        f"{'SSIM':>15}"
        f"{'Latency (ms)':>18}"
    )

    print("-" * 85)

    for name, values in summary.items():

        print(
            f"{name:<20}"
            f"{values['average_psnr_db']:>15.4f}"
            f"{values['average_ssim']:>15.4f}"
            f"{values['average_latency_ms']:>18.3f}"
        )

    print("=" * 85)
    print("CSV:", CSV_PATH)
    print("JSON:", JSON_PATH)


if __name__ == "__main__":
    evaluate()
