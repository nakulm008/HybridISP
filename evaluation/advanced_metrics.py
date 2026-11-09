
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import lpips

from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

# --------------------------------------------------
# PROJECT PATHS
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

CHECKPOINT_PATH = PROJECT_DIR / "models" / "best_model.pth"

# --------------------------------------------------
# PROJECT IMPORTS
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

LPIPS_SIZE = 512
TASK = "sidd"

PROCESS = psutil.Process()


# --------------------------------------------------
# IMAGE UTILITIES
# --------------------------------------------------

def load_rgb(path):
    image = Image.open(path).convert("RGB")

    return np.asarray(
        image,
        dtype=np.float32
    ) / 255.0


def calculate_psnr_ssim(prediction, reference):

    prediction = np.clip(prediction, 0, 1)
    reference = np.clip(reference, 0, 1)

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


def calculate_delta_e(prediction, reference):
    """
    Calculate CIEDE2000 colour difference.

    Both images must be RGB float images in [0, 1].
    """

    prediction = np.clip(prediction, 0, 1)
    reference = np.clip(reference, 0, 1)

    prediction_lab = rgb2lab(prediction)
    reference_lab = rgb2lab(reference)

    delta = deltaE_ciede2000(
        reference_lab,
        prediction_lab
    )

    return {
        "mean": float(np.mean(delta)),
        "median": float(np.median(delta)),
        "p95": float(np.percentile(delta, 95))
    }


# --------------------------------------------------
# LPIPS
# --------------------------------------------------

def prepare_lpips_tensor(image):

    resized = Image.fromarray(
        np.uint8(np.clip(image, 0, 1) * 255)
    ).resize(
        (LPIPS_SIZE, LPIPS_SIZE),
        Image.Resampling.BILINEAR
    )

    array = np.asarray(
        resized,
        dtype=np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        array.transpose(2, 0, 1).copy()
    ).unsqueeze(0)

    # LPIPS expects image values in [-1, 1].
    tensor = tensor * 2.0 - 1.0

    return tensor.to(DEVICE)


def calculate_lpips(lpips_model, prediction, reference):

    pred_tensor = prepare_lpips_tensor(prediction)
    ref_tensor = prepare_lpips_tensor(reference)

    with torch.inference_mode():

        score = lpips_model(
            pred_tensor,
            ref_tensor
        )

    return float(score.item())


# --------------------------------------------------
# SYSTEM INFORMATION
# --------------------------------------------------

def get_memory_mb():
    return PROCESS.memory_info().rss / (1024 ** 2)


def get_model_information():

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    checkpoint_size_mb = (
        CHECKPOINT_PATH.stat().st_size / (1024 ** 2)
        if CHECKPOINT_PATH.exists()
        else None
    )

    return {
        "parameters": int(parameters),
        "checkpoint_size_mb": (
            round(checkpoint_size_mb, 3)
            if checkpoint_size_mb is not None
            else None
        )
    }


# --------------------------------------------------
# INFERENCE
# --------------------------------------------------

def run_inference(noisy):

    outputs = {}

    # Classical
    start_memory = get_memory_mb()
    start = time.perf_counter()

    outputs["classical"] = classical_isp(noisy)

    classical_latency = (
        time.perf_counter() - start
    ) * 1000

    classical_memory = get_memory_mb() - start_memory

    # Learned
    start_memory = get_memory_mb()
    start = time.perf_counter()

    outputs["learned"] = learned_isp(
        noisy,
        task=TASK,
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        batch_size=BATCH_SIZE
    )

    learned_latency = (
        time.perf_counter() - start
    ) * 1000

    learned_memory = get_memory_mb() - start_memory

    # Hybrid
    start_memory = get_memory_mb()
    start = time.perf_counter()

    outputs["hybrid"] = hybrid_isp(
        noisy,
        task=TASK,
        learned_weight=0.85
    )

    hybrid_latency = (
        time.perf_counter() - start
    ) * 1000

    hybrid_memory = get_memory_mb() - start_memory

    performance = {
        "classical": {
            "latency_ms": classical_latency,
            "rss_change_mb": classical_memory
        },
        "learned": {
            "latency_ms": learned_latency,
            "rss_change_mb": learned_memory
        },
        "hybrid": {
            "latency_ms": hybrid_latency,
            "rss_change_mb": hybrid_memory
        }
    }

    return outputs, performance


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate_method(
    name,
    prediction,
    reference,
    lpips_model
):

    psnr, ssim = calculate_psnr_ssim(
        prediction,
        reference
    )

    delta = calculate_delta_e(
        prediction,
        reference
    )

    lpips_score = calculate_lpips(
        lpips_model,
        prediction,
        reference
    )

    return {
        "method": name,
        "psnr": psnr,
        "ssim": ssim,
        "delta_e_mean": delta["mean"],
        "delta_e_median": delta["median"],
        "delta_e_p95": delta["p95"],
        "lpips": lpips_score
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Advanced HybridISP evaluation."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of SIDD pairs to evaluate. 0 means all."
    )

    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be zero or positive.")

    print("=" * 70)
    print("HYBRIDISP — ADVANCED EVALUATION")
    print("=" * 70)

    print("Device:", DEVICE)
    print("Tile size:", TILE_SIZE)
    print("Overlap:", OVERLAP)
    print("Batch size:", BATCH_SIZE)

    if not DATA_DIR.exists():
        raise FileNotFoundError(DATA_DIR)

    noisy_files = sorted(
        DATA_DIR.rglob("NOISY_SRGB_010.PNG")
    )

    pairs = []

    for noisy_path in noisy_files:

        gt_path = noisy_path.with_name(
            "GT_SRGB_010.PNG"
        )

        if gt_path.exists():
            pairs.append((noisy_path, gt_path))

    if args.limit > 0:
        pairs = pairs[:args.limit]

    if not pairs:
        raise RuntimeError("No valid SIDD pairs found.")

    print("Pairs selected:", len(pairs))

    # Load pretrained LPIPS AlexNet.
    print("\nLoading LPIPS AlexNet...")

    lpips_model = lpips.LPIPS(
        net="alex",
        verbose=False
    ).to(DEVICE)

    lpips_model.eval()

    print("LPIPS model loaded.")

    model_info = get_model_information()

    print("\nModel parameters:", model_info["parameters"])
    print(
        "Checkpoint size:",
        model_info["checkpoint_size_mb"],
        "MB"
    )

    print(
        "Initial process RSS:",
        round(get_memory_mb(), 2),
        "MB"
    )

    all_results = []

    for index, (noisy_path, gt_path) in enumerate(
        pairs,
        start=1
    ):

        print(
            f"\n[{index}/{len(pairs)}] "
            f"{noisy_path.parent.name}",
            flush=True
        )

        noisy = load_rgb(noisy_path)
        reference = load_rgb(gt_path)

        if noisy.shape != reference.shape:
            raise ValueError(
                f"Dimension mismatch: {noisy_path}"
            )

        outputs, performance = run_inference(noisy)

        # Original input.
        outputs["original"] = noisy

        for method in [
            "original",
            "classical",
            "learned",
            "hybrid"
        ]:

            print(
                f"Calculating {method} metrics...",
                flush=True
            )

            result = evaluate_method(
                method,
                outputs[method],
                reference,
                lpips_model
            )

            result["image"] = noisy_path.name
            result["scene"] = noisy_path.parent.name

            if method in performance:
                result.update(performance[method])
            else:
                result["latency_ms"] = 0.0
                result["rss_change_mb"] = 0.0

            all_results.append(result)

            print(
                f"PSNR={result['psnr']:.4f}, "
                f"SSIM={result['ssim']:.4f}, "
                f"DeltaE={result['delta_e_mean']:.4f}, "
                f"LPIPS={result['lpips']:.4f}"
            )

        print(
            "Current process RSS:",
            round(get_memory_mb(), 2),
            "MB"
        )

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    df = pd.DataFrame(all_results)

    csv_path = (
        RESULTS_DIR
        / f"advanced_evaluation_{len(pairs)}_pairs.csv"
    )

    df.to_csv(csv_path, index=False)

    metric_columns = [
        "psnr",
        "ssim",
        "delta_e_mean",
        "delta_e_median",
        "delta_e_p95",
        "lpips",
        "latency_ms",
        "rss_change_mb"
    ]

    summary = (
        df.groupby("method")[metric_columns]
        .mean()
        .round(5)
    )

    summary_path = (
        RESULTS_DIR
        / f"advanced_summary_{len(pairs)}_pairs.json"
    )

    result_json = {
        "dataset": "SIDD Small sRGB",
        "pairs_evaluated": len(pairs),
        "device": str(DEVICE),
        "model": model_info,
        "configuration": {
            "tile_size": TILE_SIZE,
            "overlap": OVERLAP,
            "batch_size": BATCH_SIZE,
            "lpips_network": "AlexNet",
            "lpips_evaluation_size": LPIPS_SIZE
        },
        "process_rss_after_evaluation_mb": round(
            get_memory_mb(),
            3
        ),
        "mean_metrics": summary.to_dict(orient="index")
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result_json,
            file,
            indent=4
        )

    # --------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 100)
    print("ADVANCED EVALUATION SUMMARY")
    print("=" * 100)

    print(summary.to_string())

    print("\nCSV:", csv_path)
    print("JSON:", summary_path)

    print("\nAdvanced evaluation completed.")


if __name__ == "__main__":
    main()
