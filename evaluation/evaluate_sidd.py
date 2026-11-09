
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "SIDD_Small_sRGB_Only"
    / "Data"
)

RESULTS_DIR = PROJECT_DIR / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# ISP IMPORTS
# --------------------------------------------------

from models.model_loader import model, DEVICE
from isp.classical_isp import classical_isp
from isp.learned_isp import learned_isp
from isp.hybrid_isp import hybrid_isp


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

TILE_SIZE = 256
OVERLAP = 32
BATCH_SIZE = 1

TASK = "sidd"


# --------------------------------------------------
# IMAGE UTILITIES
# --------------------------------------------------

def load_rgb_image(path):
    """Load an RGB image and normalize it to [0, 1]."""

    image = Image.open(path).convert("RGB")

    return np.asarray(
        image,
        dtype=np.float32
    ) / 255.0


def calculate_metrics(prediction, reference):
    """Calculate PSNR and SSIM."""

    prediction = np.clip(
        prediction.astype(np.float32),
        0.0,
        1.0
    )

    reference = np.clip(
        reference.astype(np.float32),
        0.0,
        1.0
    )

    if prediction.shape != reference.shape:
        raise ValueError(
            f"Image shape mismatch: "
            f"{prediction.shape} vs {reference.shape}"
        )

    psnr = peak_signal_noise_ratio(
        reference,
        prediction,
        data_range=1.0
    )

    ssim = structural_similarity(
        reference,
        prediction,
        channel_axis=-1,
        data_range=1.0
    )

    return float(psnr), float(ssim)


def measure_latency(function):
    """Execute a function and return output and elapsed milliseconds."""

    start = time.perf_counter()

    output = function()

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    return output, elapsed_ms


# --------------------------------------------------
# DATASET DISCOVERY
# --------------------------------------------------

def discover_sidd_pairs():
    """Find matching noisy and ground-truth SIDD images."""

    noisy_files = sorted(
        DATA_DIR.rglob("NOISY_SRGB_010.PNG")
    )

    pairs = []

    for noisy_path in noisy_files:

        gt_path = noisy_path.with_name(
            "GT_SRGB_010.PNG"
        )

        if not gt_path.exists():
            print(
                f"WARNING: Missing GT image: {gt_path}"
            )
            continue

        pairs.append(
            (noisy_path, gt_path)
        )

    return pairs


# --------------------------------------------------
# MODEL WARMUP
# --------------------------------------------------

def warmup_model():
    """Run a small inference before timing full images."""

    print("Warming up model...", flush=True)

    dummy_image = np.random.rand(
        128,
        128,
        3
    ).astype(np.float32)

    learned_isp(
        dummy_image,
        task=TASK,
        tile_size=128,
        overlap=32,
        batch_size=1
    )

    print("Warmup completed.\n", flush=True)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate_pair(noisy_path, gt_path, index, total):

    print(
        f"\n[{index}/{total}] {noisy_path.parent.name}/"
        f"{noisy_path.name}",
        flush=True
    )

    noisy = load_rgb_image(noisy_path)
    ground_truth = load_rgb_image(gt_path)

    if noisy.shape != ground_truth.shape:
        raise ValueError(
            f"Dimension mismatch for {noisy_path.name}: "
            f"{noisy.shape} vs {ground_truth.shape}"
        )

    # Original noisy input metrics.
    original_psnr, original_ssim = calculate_metrics(
        noisy,
        ground_truth
    )

    # Classical ISP.
    classical_output, classical_latency = measure_latency(
        lambda: classical_isp(noisy)
    )

    classical_psnr, classical_ssim = calculate_metrics(
        classical_output,
        ground_truth
    )

    # Learned ISP.
    learned_output, learned_latency = measure_latency(
        lambda: learned_isp(
            noisy,
            task=TASK,
            tile_size=TILE_SIZE,
            overlap=OVERLAP,
            batch_size=BATCH_SIZE
        )
    )

    learned_psnr, learned_ssim = calculate_metrics(
        learned_output,
        ground_truth
    )

    # Hybrid ISP.
    hybrid_output, hybrid_latency = measure_latency(
        lambda: hybrid_isp(
            noisy,
            task=TASK
        )
    )

    hybrid_psnr, hybrid_ssim = calculate_metrics(
        hybrid_output,
        ground_truth
    )

    print(
        f"Classical: {classical_latency:.2f} ms"
    )

    print(
        f"Learned:   {learned_latency:.2f} ms"
    )

    print(
        f"Hybrid:    {hybrid_latency:.2f} ms"
    )

    return {
        "image": noisy_path.name,
        "scene": noisy_path.parent.name,

        "original_psnr": original_psnr,
        "original_ssim": original_ssim,

        "classical_psnr": classical_psnr,
        "classical_ssim": classical_ssim,
        "classical_latency_ms": classical_latency,

        "learned_psnr": learned_psnr,
        "learned_ssim": learned_ssim,
        "learned_latency_ms": learned_latency,

        "hybrid_psnr": hybrid_psnr,
        "hybrid_ssim": hybrid_ssim,
        "hybrid_latency_ms": hybrid_latency
    }


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

def print_summary(results):

    df = pd.DataFrame(results)

    methods = [
        ("Original", "original"),
        ("Classical", "classical"),
        ("Learned", "learned"),
        ("Hybrid", "hybrid")
    ]

    print("\n" + "=" * 75)
    print("EVALUATION SUMMARY")
    print("=" * 75)

    print(
        f"{'Method':<15}"
        f"{'PSNR':>13}"
        f"{'SSIM':>13}"
        f"{'Latency (ms)':>18}"
    )

    for label, prefix in methods:

        psnr = df[f"{prefix}_psnr"].mean()
        ssim = df[f"{prefix}_ssim"].mean()

        latency_column = f"{prefix}_latency_ms"

        if latency_column in df.columns:
            latency = df[latency_column].mean()
        else:
            latency = 0.0

        print(
            f"{label:<15}"
            f"{psnr:>13.4f}"
            f"{ssim:>13.4f}"
            f"{latency:>18.3f}"
        )

    print("=" * 75)

    return df


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Evaluate HybridISP on the SIDD dataset."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of image pairs to evaluate. Use 0 for all."
    )

    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be zero or a positive integer.")

    print("=" * 65)
    print("HYBRIDISP — SIDD EVALUATION")
    print("=" * 65)

    print(f"Execution device: {DEVICE}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(
        f"Tile configuration: "
        f"{TILE_SIZE}x{TILE_SIZE}, "
        f"overlap={OVERLAP}, "
        f"batch={BATCH_SIZE}"
    )

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"SIDD dataset directory not found: {DATA_DIR}"
        )

    pairs = discover_sidd_pairs()

    if not pairs:
        raise RuntimeError(
            "No valid SIDD noisy/ground-truth pairs found."
        )

    if args.limit > 0:
        pairs = pairs[:args.limit]

    print(f"Pairs selected: {len(pairs)}")

    warmup_model()

    results = []

    for index, (noisy_path, gt_path) in enumerate(
        pairs,
        start=1
    ):

        result = evaluate_pair(
            noisy_path,
            gt_path,
            index,
            len(pairs)
        )

        results.append(result)

    df = print_summary(results)

    # --------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------

    count = len(results)

    csv_path = (
        RESULTS_DIR
        / f"sidd_evaluation_{count}_pairs.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------

    summary = {
        "dataset": "SIDD Small sRGB",
        "pairs_evaluated": count,
        "device": str(DEVICE),

        "model_parameters": sum(
            p.numel()
            for p in model.parameters()
        ),

        "configuration": {
            "tile_size": TILE_SIZE,
            "overlap": OVERLAP,
            "batch_size": BATCH_SIZE
        },

        "metrics": {
            "original": {
                "psnr": float(df["original_psnr"].mean()),
                "ssim": float(df["original_ssim"].mean())
            },

            "classical": {
                "psnr": float(df["classical_psnr"].mean()),
                "ssim": float(df["classical_ssim"].mean()),
                "latency_ms": float(
                    df["classical_latency_ms"].mean()
                )
            },

            "learned": {
                "psnr": float(df["learned_psnr"].mean()),
                "ssim": float(df["learned_ssim"].mean()),
                "latency_ms": float(
                    df["learned_latency_ms"].mean()
                )
            },

            "hybrid": {
                "psnr": float(df["hybrid_psnr"].mean()),
                "ssim": float(df["hybrid_ssim"].mean()),
                "latency_ms": float(
                    df["hybrid_latency_ms"].mean()
                )
            }
        }
    }

    json_path = (
        RESULTS_DIR
        / f"sidd_summary_{count}_pairs.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print(f"\nCSV: {csv_path}")
    print(f"JSON: {json_path}")

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
