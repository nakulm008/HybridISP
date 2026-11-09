
# HybridISP — Technical Design Document

**Problem Statement:** P2 — Intelligent and Computationally Efficient Smartphone ISP  
**Team ID:** T90

## 1. Introduction

HybridISP is an image processing system designed to combine classical computer vision techniques with deep learning-based image enhancement.

Traditional image processing methods are computationally simple and predictable but may have limitations when handling complex noise and image degradation.

Deep learning models can learn more complex image restoration patterns but require additional computational resources.

HybridISP combines both approaches to investigate the balance between image quality and computational efficiency.

## 2. Objectives

The primary objectives are:

1. Implement a conventional image processing baseline.
2. Develop a learned image enhancement pipeline.
3. Combine classical and learned outputs using weighted fusion.
4. Support different image processing tasks.
5. Evaluate image quality using quantitative metrics.
6. Measure model complexity and inference latency.
7. Provide a web-based interface for image processing and visualization.

## 3. Functional Requirements

| ID | Requirement | Description |
|---|---|---|
| FR-01 | Image Upload | Accept RGB image uploads |
| FR-02 | Input Validation | Validate uploaded image files |
| FR-03 | Classical Processing | Apply bilateral filtering |
| FR-04 | Learned Processing | Execute HybridISPNet inference |
| FR-05 | Hybrid Processing | Fuse classical and learned outputs |
| FR-06 | Task Selection | Support SIDD and FiveK processing tasks |
| FR-07 | Result Display | Display processed images |
| FR-08 | Evaluation | Calculate supported image quality metrics |

## 4. Non-Functional Requirements

- Maintain modular and understandable source code.
- Support high-resolution image processing through tiled inference.
- Avoid unnecessary memory usage during inference.
- Provide consistent output dimensions.
- Support containerized deployment.
- Maintain reproducible evaluation procedures.

## 5. System Design

The system consists of five major modules.

### 5.1 Web Application Module

**File:** `app.py`

Responsibilities:

- Receive image uploads.
- Validate input.
- Handle processing requests.
- Select the processing pipeline.
- Return processed images and evaluation information.

**Technology:** Flask.

### 5.2 Classical ISP Module

**File:** `isp/classical_isp.py`

The classical pipeline uses a bilateral filter.

A bilateral filter considers both spatial distance and intensity similarity when smoothing an image.

Its general formulation is:

\[
I'(p)=\frac{1}{W_p}
\sum_{q\in\Omega}
G_s(\|p-q\|)
G_r(|I(p)-I(q)|)I(q)
\]

Where:

- \(I(p)\) is the original pixel value.
- \(I'(p)\) is the filtered pixel value.
- \(G_s\) represents spatial similarity.
- \(G_r\) represents intensity similarity.
- \(W_p\) is the normalization factor.

The implementation uses OpenCV bilateral filtering with:

- Diameter: 7
- Sigma Color: 35
- Sigma Space: 35

The classical pipeline serves as a conventional image-processing baseline.

### 5.3 Learned ISP Module

**Files:**
- `models/hybrid_isp_model.py`
- `models/model_loader.py`
- `isp/learned_isp.py`

The learned pipeline uses a custom convolutional neural network called HybridISPNet.

Model characteristics:

- Framework: PyTorch.
- Parameters: approximately 3,086,051.
- Checkpoint: `models/best_model.pth`.
- Training tasks: SIDD denoising and FiveK image enhancement.

The model uses task conditioning to distinguish between the supported processing objectives.

The trained checkpoint is loaded through the model loader and used for inference.

#### Tiled Inference

High-resolution images are divided into smaller overlapping tiles.

Current configuration:

| Parameter | Value |
|---|---:|
| Tile size | 256 × 256 |
| Overlap | 32 pixels |
| Batch size | 1 |

Overlapping predictions are combined using weighted aggregation to reduce visible boundaries between tiles.

Tiled inference helps manage per-tile memory requirements, although it introduces additional processing overhead.

### 5.4 Hybrid ISP Module

**File:** `isp/hybrid_isp.py`

The hybrid module combines the output of the learned and classical processing branches.

The current default fusion equation is:

\[
I_{hybrid} =
\alpha I_{learned}+
(1-\alpha)I_{classical}
\]

The current default value is:

\[
\alpha=0.85
\]

Therefore:

\[
I_{hybrid} =
0.85I_{learned}+0.15I_{classical}
\]

The final image is clipped to the valid normalized pixel range [0, 1].

The fusion weight is configurable, allowing different combinations to be evaluated.

### 5.5 Evaluation Module

**Files:** `isp/metrics.py` and `evaluation/`

The evaluation module supports several image quality and computational measurements.

#### PSNR

Peak Signal-to-Noise Ratio measures reconstruction error.

Higher values generally indicate lower pixel-level reconstruction error.

#### SSIM

Structural Similarity Index compares image structure, luminance, and contrast.

Higher values indicate greater structural similarity.

#### Delta E 2000

Measures perceptual color difference between an output and its reference image.

Lower values indicate smaller color differences.

#### LPIPS

Learned Perceptual Image Patch Similarity measures perceptual differences using deep feature representations.

Lower values generally indicate greater perceptual similarity.

#### Computational Measurements

The project also records:

- Model parameter count.
- Checkpoint size.
- Inference latency.
- Process memory measurements where available.

Metrics requiring a reference image are calculated using corresponding ground-truth images.

## 6. Dataset and Training Design

Two datasets were used during model training.

### 6.1 SIDD

Smartphone Image Denoising Dataset.

Purpose:

- Train the model for smartphone image denoising.
- Learn to reduce noise while retaining image details.

### 6.2 MIT-Adobe FiveK

Purpose:

- Train the model for image enhancement.
- Learn image appearance transformations using paired input and reference images.

### 6.3 Training Configuration

| Parameter | Configuration |
|---|---|
| Framework | PyTorch |
| Training environment | Google Colab |
| GPU | NVIDIA T4 |
| Patch size | 128 × 128 |
| Base channels | 32 |
| Tasks | 2 |
| Epochs | 5 |
| Steps per epoch | 50 |
| Batch size | 8 |
| Learning rate | 0.0002 |
| Optimizer | AdamW |

The training configuration represents a limited experimental training run, not a fully converged production model.

## 7. Technology Selection

| Technology | Reason for Selection |
|---|---|
| Python | Main implementation language |
| PyTorch | Neural network training and inference |
| OpenCV | Classical image processing |
| NumPy | Numerical image operations |
| Flask | Lightweight web application |
| Scikit-image | Image quality evaluation |
| LPIPS | Perceptual quality evaluation |
| Docker | Application containerization |

## 8. Computational Efficiency Considerations

The system investigates computational efficiency through:

1. A relatively compact neural network.
2. Tiled inference for high-resolution inputs.
3. Configurable tile sizes and overlap.
4. Batch-size experiments.
5. Model parameter and checkpoint-size measurements.
6. CPU inference benchmarking.

The current implementation has not yet demonstrated quantization or pruning.

The CPU inference time for high-resolution images remains a significant limitation.

## 9. Testing Strategy

The project includes testing and evaluation scripts.

| Test | Purpose |
|---|---|
| `test_model.py` | Verify model loading |
| `test_isp.py` | Verify image processing pipelines |
| `test_metrics.py` | Verify metric calculations |
| `evaluation/verify_sidd.py` | Verify SIDD image pairs |
| `evaluation/evaluate_sidd.py` | Compare processing approaches |
| `evaluation/advanced_metrics.py` | Evaluate additional metrics |
| `evaluation/ablation_study.py` | Compare fusion weights |
| `evaluation/benchmark_tiles.py` | Compare inference tile configurations |

Evaluation results currently cover limited samples and should be treated as preliminary.

## 10. Known Limitations

1. The available datasets contain sRGB images rather than RAW Bayer sensor data.
2. The implementation does not currently demonstrate a complete RAW-to-RGB ISP pipeline.
3. The classical pipeline currently uses bilateral filtering rather than a complete traditional camera ISP.
4. High-resolution CPU inference is slow.
5. The training run is limited to five epochs and requires further experimentation.
6. The current evaluation samples are insufficient to establish generalization across the complete dataset.
7. Quantization and pruning have not yet been implemented.
8. The current hybrid fusion weight requires broader evaluation before claiming an optimal value.

## 11. Future Improvements

Potential improvements include:

- Training with a larger number of epochs.
- Evaluating on the full SIDD test set.
- Adding additional classical ISP stages.
- Supporting RAW Bayer sensor input.
- Implementing model quantization.
- Investigating structured pruning.
- Optimizing inference for mobile hardware.
- Expanding cross-dataset evaluation.

## 12. Conclusion

HybridISP provides a modular framework for investigating the combination of classical image processing and deep learning-based image enhancement.

Its three processing approaches allow direct comparison between conventional filtering, learned processing, and hybrid fusion.

The project includes a Flask application, trained neural network, evaluation scripts, and Docker configuration.

Further training, broader evaluation, and deployment optimization are required before the system can be considered production-ready.
