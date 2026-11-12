
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request, jsonify, url_for

from models.model_loader import model, DEVICE
from isp.classical_isp import classical_isp
from isp.learned_isp import learned_isp
from isp.hybrid_isp import fuse_outputs
from isp.metrics import evaluate_all_outputs


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def load_rgb_image(file_storage):
    image = Image.open(file_storage.stream).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def save_rgb_image(array, path):
    array = np.clip(array, 0.0, 1.0)

    image = Image.fromarray(
        (array * 255).round().astype(np.uint8),
        mode="RGB"
    )

    image.save(path, format="PNG")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def process_image():

    input_file = request.files.get("image")
    reference_file = request.files.get("reference")

    if input_file is None or not input_file.filename:
        return jsonify({
            "error": "Please select an input image."
        }), 400

    if not allowed_file(input_file.filename):
        return jsonify({
            "error": "Input must be a PNG, JPG or JPEG image."
        }), 400

    if reference_file and reference_file.filename:
        if not allowed_file(reference_file.filename):
            return jsonify({
                "error": "Reference must be a PNG, JPG or JPEG image."
            }), 400

    task = request.form.get("task", "sidd").lower()

    if task not in {"sidd", "fivek"}:
        return jsonify({
            "error": "Invalid processing profile."
        }), 400

    file_id = uuid.uuid4().hex

    try:
        # Load input image
        original = load_rgb_image(input_file)

        # Load optional reference image
        reference = None

        if reference_file and reference_file.filename:
            reference = load_rgb_image(reference_file)

            if reference.shape != original.shape:
                return jsonify({
                    "error": (
                        "Input and reference dimensions must match. "
                        f"Input: {original.shape[1]}x{original.shape[0]}, "
                        f"Reference: {reference.shape[1]}x{reference.shape[0]}."
                    )
                }), 400

        height, width = original.shape[:2]

        # Branch 1: Classical ISP
        start = time.perf_counter()
        classical = classical_isp(original)
        classical_time = (
            time.perf_counter() - start
        ) * 1000

        # Branch 2: Learned ISP
        start = time.perf_counter()
        learned = learned_isp(original, task=task)
        learned_time = (
            time.perf_counter() - start
        ) * 1000

        # Branch 3: Hybrid fusion
        # Reuse the two outputs above. No repeated model inference.
        start = time.perf_counter()

        hybrid = fuse_outputs(
            original,
            learned,
            classical,
            task=task,
            learned_weight=0.15,
            adaptive=False
        )

        fusion_time = (
            time.perf_counter() - start
        ) * 1000

        # Total hybrid pipeline latency includes both branches and fusion.
        hybrid_time = (
            classical_time + learned_time + fusion_time
        )

        outputs = {
            "original": original,
            "classical": classical,
            "learned": learned,
            "hybrid": hybrid
        }

        # Save result images
        result_urls = {}

        for name, output in outputs.items():
            output_path = RESULT_DIR / f"{file_id}_{name}.png"

            save_rgb_image(output, output_path)

            result_urls[name] = url_for(
                "static",
                filename=f"results/{file_id}_{name}.png"
            )

        # Model information
        parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        checkpoint_path = BASE_DIR / "models" / "best_model.pth"

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {checkpoint_path}"
            )

        model_size_mb = (
            checkpoint_path.stat().st_size / (1024 * 1024)
        )

        # Runtime metrics
        metrics = {
            "classical_latency_ms": round(classical_time, 2),
            "learned_latency_ms": round(learned_time, 2),
            "hybrid_latency_ms": round(hybrid_time, 2),
            "fusion_latency_ms": round(fusion_time, 2),
            "model_parameters": parameters,
            "model_size_mb": round(model_size_mb, 2),
            "image_resolution": f"{width} x {height}",
            "device": str(DEVICE)
        }

        # Quality evaluation only when reference is supplied
        quality_metrics = None

        if reference is not None:
            quality_metrics = evaluate_all_outputs(
                outputs,
                reference
            )

        return jsonify({
            "success": True,
            "message": "Image processing completed.",
            "results": result_urls,
            "metrics": metrics,
            "quality_metrics": quality_metrics,
            "reference_provided": reference is not None
        })

    except (UnidentifiedImageError, OSError, ValueError) as error:
        app.logger.exception(
            "Image processing or evaluation error"
        )

        return jsonify({
            "error": str(error)
        }), 400

    except Exception as error:
        app.logger.exception(
            "Unexpected processing error"
        )

        return jsonify({
            "error": f"Processing failed: {str(error)}"
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
