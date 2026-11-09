
import csv
import json
import statistics
import sys
import time
from pathlib import Path

# Resolve the project root before importing project modules.
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import torch
from PIL import Image

from models.model_loader import DEVICE
from isp.classical_isp import classical_isp
from isp.learned_isp import learned_isp
from isp.hybrid_isp import hybrid_isp
from isp.metrics import calculate_metrics


DATASET_DIR = (
    PROJECT_DIR / "data" / "SIDD_Small_sRGB_Only" / "Data"
)

OUTPUT_DIR = PROJECT_DIR / "evaluation" / "results"
CSV_PATH = OUTPUT_DIR / "sidd_evaluation.csv"
JSON_PATH = OUTPUT_DIR / "sidd_summary.json"

TASK = "sidd"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(
            image.convert("RGB"),
            dtype=np.float32
        ) / 255.0


def sync_device():
    if str(DEVICE).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_latency(function):
    sync_device()
    start = time.perf_counter()
    output = function()
    sync_device()
    elapsed = (time.perf_counter() - start) * 1000
    return output, elapsed


def discover_sidd_pairs():

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"SIDD dataset not found: {DATASET_DIR}"
        )

    pairs = []

    for scene_dir in sorted(DATASET_DIR.iterdir()):

        if not scene_dir.is_dir():
            continue

        for noisy_path in sorted(
            scene_dir.glob("NOISY_SRGB_*.PNG")
        ):

            suffix = noisy_path.name.replace(
                "NOISY_SRGB_", ""
            )

            gt_path = scene_dir / f"GT_SRGB_{suffix}"

            if gt_path.exists():
                pairs.append((noisy_path, gt_path))

    return pairs


def evaluate_dataset():

    pairs = discover_sidd_pairs()

    if not pairs:
        raise RuntimeError(
            f"No valid image pairs found in {DATASET_DIR}"
        )

    print("=" * 70)
    print("HYBRIDISP — SIDD DATASET EVALUATION")
    print("=" * 70)
    print("Project root:", PROJECT_DIR)
    print("Dataset pairs:", len(pairs))
    print("Execution device:", DEVICE)

    # Warmup
    print("\nWarming up model...")
    first_image = load_rgb(pairs[0][0])
    learned_isp(first_image, task=TASK)
    sync_device()
    print("Warmup completed.")

    records = []

    methods = [
        "original",
        "classical",
        "learned",
        "hybrid"
    ]

    for index, (noisy_path, gt_path) in enumerate(
        pairs, start=1
    ):

        print(
            f"\n[{index}/{len(pairs)}] "
            f"{noisy_path.parent.name}/{noisy_path.name}"
        )

        noisy = load_rgb(noisy_path)
        reference = load_rgb(gt_path)

        if noisy.shape != reference.shape:
            print("Skipped: dimension mismatch.")
            continue

        classical, classical_time = measure_latency(
            lambda: classical_isp(noisy)
        )

        learned, learned_time = measure_latency(
            lambda: learned_isp(noisy, task=TASK)
        )

        hybrid, hybrid_time = measure_latency(
            lambda: hybrid_isp(noisy, task=TASK)
        )

        outputs = {
            "original": noisy,
            "classical": classical,
            "learned": learned,
            "hybrid": hybrid
        }

        latencies = {
            "original": 0.0,
            "classical": classical_time,
            "learned": learned_time,
            "hybrid": hybrid_time
        }

        for method in methods:

            quality = calculate_metrics(
                outputs[method],
                reference
            )

            records.append({
                "scene": noisy_path.parent.name,
                "image": noisy_path.name,
                "method": method,
                "psnr_db": quality["psnr"],
                "ssim": quality["ssim"],
                "latency_ms": round(
                    latencies[method], 3
                )
            })

        print(f"Classical: {classical_time:.2f} ms")
        print(f"Learned:   {learned_time:.2f} ms")
        print(f"Hybrid:    {hybrid_time:.2f} ms")

    if not records:
        raise RuntimeError("No images were evaluated successfully.")

    # CSV export
    fieldnames = [
        "scene",
        "image",
        "method",
        "psnr_db",
        "ssim",
        "latency_ms"
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(records)

    # Aggregate results
    summary = {}

    for method in methods:

        rows = [
            row for row in records
            if row["method"] == method
        ]

        summary[method] = {
            "image_count": len(rows),
            "average_psnr_db": round(
                statistics.mean(
                    row["psnr_db"] for row in rows
                ), 4
            ),
            "average_ssim": round(
                statistics.mean(
                    row["ssim"] for row in rows
                ), 4
            ),
            "average_latency_ms": round(
                statistics.mean(
                    row["latency_ms"] for row in rows
                ), 3
            )
        }

    summary_data = {
        "experiment": "HybridISP SIDD Evaluation",
        "dataset": "SIDD Small sRGB",
        "execution_device": str(DEVICE),
        "successfully_evaluated_pairs": len({
            (row["scene"], row["image"])
            for row in records
        }),
        "results": summary
    }

    with open(
        JSON_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(summary_data, file, indent=4)

    # Print results
    print("\n" + "=" * 75)
    print("FINAL SIDD EVALUATION")
    print("=" * 75)

    print(
        f"{'Method':<15}"
        f"{'PSNR (dB)':>15}"
        f"{'SSIM':>15}"
        f"{'Latency (ms)':>18}"
    )

    print("-" * 75)

    for method, values in summary.items():

        print(
            f"{method:<15}"
            f"{values['average_psnr_db']:>15.4f}"
            f"{values['average_ssim']:>15.4f}"
            f"{values['average_latency_ms']:>18.3f}"
        )

    print("=" * 75)
    print("CSV:", CSV_PATH)
    print("JSON:", JSON_PATH)
    print("Evaluation completed.")


if __name__ == "__main__":
    evaluate_dataset()
