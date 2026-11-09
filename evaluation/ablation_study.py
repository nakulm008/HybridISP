
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
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

# --------------------------------------------------
# PROJECT IMPORTS
# --------------------------------------------------

from models.model_loader import DEVICE
from isp.classical_isp import classical_isp
from isp.learned_isp import learned_isp

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

TILE_SIZE = 256
OVERLAP = 32
BATCH_SIZE = 1

FUSION_WEIGHTS = [0.70, 0.85, 0.90, 1.00]

LPIPS_SIZE = 512
TASK = "sidd"


# --------------------------------------------------
# IMAGE UTILITIES
# --------------------------------------------------

def load_rgb(path):

    image = Image.open(path).convert("RGB")

    return np.asarray(
        image,
        dtype=np.float32
    ) / 255.0


def calculate_metrics(prediction, reference, lpips_model):

    prediction = np.clip(prediction, 0.0, 1.0)
    reference = np.clip(reference, 0.0, 1.0)

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

    prediction_lab = rgb2lab(prediction)
    reference_lab = rgb2lab(reference)

    delta_e = deltaE_ciede2000(
        reference_lab,
        prediction_lab
    )

    # LPIPS uses resized 512 x 512 images.
    pred_small = Image.fromarray(
        np.uint8(prediction * 255)
    ).resize(
        (LPIPS_SIZE, LPIPS_SIZE),
        Image.Resampling.BILINEAR
    )

    ref_small = Image.fromarray(
        np.uint8(reference * 255)
    ).resize(
        (LPIPS_SIZE, LPIPS_SIZE),
        Image.Resampling.BILINEAR
    )

    pred_array = np.asarray(
        pred_small,
        dtype=np.float32
    ) / 255.0

    ref_array = np.asarray(
        ref_small,
        dtype=np.float32
    ) / 255.0

    pred_tensor = torch.from_numpy(
        pred_array.transpose(2, 0, 1).copy()
    ).unsqueeze(0)

    ref_tensor = torch.from_numpy(
        ref_array.transpose(2, 0, 1).copy()
    ).unsqueeze(0)

    pred_tensor = (
        pred_tensor * 2.0 - 1.0
    ).to(DEVICE)

    ref_tensor = (
        ref_tensor * 2.0 - 1.0
    ).to(DEVICE)

    with torch.inference_mode():

        lpips_score = lpips_model(
            pred_tensor,
            ref_tensor
        ).item()

    return {
        "psnr": float(psnr),
        "ssim": float(ssim),
        "delta_e_mean": float(np.mean(delta_e)),
        "lpips": float(lpips_score)
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="HybridISP fusion-weight ablation study."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of SIDD pairs. Use 0 for all."
    )

    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be zero or positive.")

    print("=" * 70)
    print("HYBRIDISP — FUSION WEIGHT ABLATION STUDY")
    print("=" * 70)

    print("Device:", DEVICE)
    print("Tile size:", TILE_SIZE)
    print("Overlap:", OVERLAP)
    print("Batch size:", BATCH_SIZE)

    # Discover image pairs.
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

    # Load LPIPS model.
    print("\nLoading LPIPS AlexNet...")

    lpips_model = lpips.LPIPS(
        net="alex",
        verbose=False
    ).to(DEVICE)

    lpips_model.eval()

    print("LPIPS loaded.")

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

        # Run each component only once.
        print("Running classical ISP...", flush=True)

        classical_output = classical_isp(noisy)

        print("Running learned ISP...", flush=True)

        learned_output = learned_isp(
            noisy,
            task=TASK,
            tile_size=TILE_SIZE,
            overlap=OVERLAP,
            batch_size=BATCH_SIZE
        )

        if classical_output.shape != learned_output.shape:
            raise ValueError(
                "Classical and learned output dimensions differ."
            )

        # Test fusion weights using the same predictions.
        for learned_weight in FUSION_WEIGHTS:

            classical_weight = 1.0 - learned_weight

            fused_output = (
                learned_weight * learned_output
                + classical_weight * classical_output
            )

            fused_output = np.clip(
                fused_output,
                0.0,
                1.0
            )

            metrics = calculate_metrics(
                fused_output,
                reference,
                lpips_model
            )

            result = {
                "image": noisy_path.name,
                "scene": noisy_path.parent.name,
                "learned_weight": learned_weight,
                "classical_weight": classical_weight,
                **metrics
            }

            all_results.append(result)

            print(
                f"Learned={learned_weight:.0%}, "
                f"Classical={classical_weight:.0%} | "
                f"PSNR={metrics['psnr']:.4f}, "
                f"SSIM={metrics['ssim']:.4f}, "
                f"DeltaE={metrics['delta_e_mean']:.4f}, "
                f"LPIPS={metrics['lpips']:.4f}",
                flush=True
            )

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    df = pd.DataFrame(all_results)

    csv_path = (
        RESULTS_DIR
        / f"ablation_{len(pairs)}_pairs.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    summary = (
        df.groupby(
            ["learned_weight", "classical_weight"]
        )[
            [
                "psnr",
                "ssim",
                "delta_e_mean",
                "lpips"
            ]
        ]
        .mean()
        .reset_index()
    )

    json_path = (
        RESULTS_DIR
        / f"ablation_summary_{len(pairs)}_pairs.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "dataset": "SIDD Small sRGB",
                "pairs_evaluated": len(pairs),
                "device": str(DEVICE),
                "configuration": {
                    "tile_size": TILE_SIZE,
                    "overlap": OVERLAP,
                    "batch_size": BATCH_SIZE
                },
                "results": summary.to_dict(
                    orient="records"
                )
            },
            file,
            indent=4
        )

    print("\n" + "=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)

    print(summary.to_string(index=False))

    print("\nCSV:", csv_path)
    print("JSON:", json_path)

    print("\nAblation study completed.")


if __name__ == "__main__":
    main()
