
import os
import csv
import time

import cv2
import lpips
import numpy as np
import torch

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.color import rgb2lab, deltaE_ciede2000

from isp.classical_isp import classical_isp
from isp.learned_isp import learned_isp
from isp.hybrid_isp import hybrid_isp


NOISY_PATH = r"C:\Users\sagar\Downloads\HybridISP_noisy_RGB_test.png"
REFERENCE_PATH = r"C:\Users\sagar\Downloads\HybridISP_clean_RGB_reference.png"

OUTPUT_DIR = "evaluation/adaptive_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_rgb(path):
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(path)

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    ).astype(np.float32) / 255.0


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
        channel_axis=2,
        data_range=1.0
    )

    pred_lab = rgb2lab(prediction)
    ref_lab = rgb2lab(reference)

    delta_e = float(
        np.mean(
            deltaE_ciede2000(
                ref_lab,
                pred_lab
            )
        )
    )

    pred_tensor = torch.from_numpy(
        prediction.transpose(2, 0, 1)
    ).unsqueeze(0).float() * 2.0 - 1.0

    ref_tensor = torch.from_numpy(
        reference.transpose(2, 0, 1)
    ).unsqueeze(0).float() * 2.0 - 1.0

    with torch.no_grad():
        lpips_value = lpips_model(
            pred_tensor,
            ref_tensor
        ).item()

    return {
        "PSNR_dB": float(psnr),
        "SSIM": float(ssim),
        "DeltaE2000": delta_e,
        "LPIPS": float(lpips_value)
    }


def save_image(name, image):
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, name + ".png"),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )


def main():
    print("Loading images...")

    noisy = load_rgb(NOISY_PATH)
    reference = load_rgb(REFERENCE_PATH)

    if noisy.shape != reference.shape:
        raise ValueError(
            f"Image dimensions differ: {noisy.shape} vs {reference.shape}"
        )

    print("Image dimensions:", noisy.shape)

    print("\nLoading LPIPS AlexNet metric...")
    lpips_model = lpips.LPIPS(net="alex")
    lpips_model.eval()

    pipelines = {}

    # 1. Noisy input baseline
    pipelines["Noisy Input"] = noisy

    # 2. Classical ISP
    print("\nRunning Classical ISP...")
    start = time.perf_counter()
    pipelines["Classical ISP"] = classical_isp(noisy)
    print("Time:", round(time.perf_counter() - start, 3), "seconds")

    # 3. Learned ISP
    print("\nRunning Learned ISP...")
    start = time.perf_counter()

    pipelines["Learned ISP"] = learned_isp(
        noisy,
        task="sidd",
        tile_size=256,
        overlap=32,
        batch_size=1
    )

    print("Time:", round(time.perf_counter() - start, 3), "seconds")

    # 4. Original fixed fusion
    print("\nRunning Fixed Hybrid ISP...")
    start = time.perf_counter()

    pipelines["Fixed Hybrid 85/15"] = hybrid_isp(
        noisy,
        task="sidd",
        learned_weight=0.85,
        adaptive=False
    )

    print("Time:", round(time.perf_counter() - start, 3), "seconds")

    # 5. New adaptive fusion
    print("\nRunning Adaptive Hybrid ISP...")
    start = time.perf_counter()

    pipelines["Adaptive Hybrid"] = hybrid_isp(
        noisy,
        task="sidd",
        learned_weight=0.85,
        adaptive=True
    )

    print("Time:", round(time.perf_counter() - start, 3), "seconds")

    results = []

    print("\nCalculating metrics...")

    for name, output in pipelines.items():
        output = np.asarray(output, dtype=np.float32)

        if output.shape != reference.shape:
            raise ValueError(
                f"{name} returned an incorrect shape: {output.shape}"
            )

        metrics = calculate_metrics(
            output,
            reference,
            lpips_model
        )

        metrics["Pipeline"] = name
        results.append(metrics)

        save_image(
            name.replace(" ", "_").replace("/", "_"),
            output
        )

        print("\n", name)
        for key, value in metrics.items():
            if key != "Pipeline":
                print(f"{key}: {value:.6f}")

    csv_path = os.path.join(
        OUTPUT_DIR,
        "adaptive_comparison.csv"
    )

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Pipeline",
                "PSNR_dB",
                "SSIM",
                "DeltaE2000",
                "LPIPS"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n===================================")
    print("EVALUATION COMPLETED")
    print("CSV:", csv_path)
    print("Images:", OUTPUT_DIR)
    print("===================================")


if __name__ == "__main__":
    main()
