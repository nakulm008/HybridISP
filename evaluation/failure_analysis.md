
# HybridISP — Failure Case Analysis

## 1. Evaluation Context

HybridISP is a hybrid smartphone image signal processing pipeline combining:
- Classical image processing using bilateral filtering.
- A task-conditioned learned ISP model.
- Weighted fusion of classical and learned outputs.

Dataset: SIDD Small sRGB  
Input resolution: 5328 × 3000  
Model parameters: 3,086,051  
Checkpoint size: 35.407 MB  
Execution device: CPU

## 2. Current Observations

The preliminary evaluation on one SIDD image produced:

| Method | PSNR | SSIM | Delta E | LPIPS |
|---|---:|---:|---:|---:|
| Original | 31.3686 | 0.6387 | 5.2525 | 0.0081 |
| Classical | 38.1520 | 0.9176 | 2.7813 | 0.0073 |
| Learned | 38.6487 | 0.9125 | 2.4569 | 0.0082 |
| Hybrid 85/15 | 39.3024 | 0.9256 | 2.2475 | 0.0071 |
| Hybrid 70/30 | 39.7487 | 0.9345 | 2.1233 | 0.0064 |

Note: The 70/30 result is from a preliminary single-image ablation, not a full-dataset result.

## 3. Identified Limitations and Failure Risks

### 3.1 High CPU Inference Latency

Learned inference takes approximately 77–92 seconds on the tested full-resolution image, depending on the run.

Impact:
- Not suitable for real-time full-resolution CPU processing.
- Slow interactive image enhancement.
- High computational cost compared with classical filtering.

Possible improvement:
- GPU acceleration.
- Quantization and optimized inference runtimes.
- Reduced-resolution preview processing.
- Further profiling and model architecture optimization.

### 3.2 Tile Boundary Artifacts

The learned model processes large images using overlapping tiles.

Potential issue:
- Adjacent patches may produce slightly different predictions.
- Incorrect blending can create visible seams.
- Edge padding may affect predictions near image boundaries.

Mitigation:
- Overlapping tiles.
- Weighted Hann-window blending.
- Testing different tile sizes and overlap values.

### 3.3 Excessive Smoothing

Bilateral filtering can remove noise but may also remove fine image details.

Potential impact:
- Loss of texture.
- Reduced sharpness.
- Fine structures may become blurred.

Mitigation:
- Adjust bilateral filter parameters.
- Combine classical and learned predictions.
- Evaluate texture preservation using visual comparisons and perceptual metrics.

### 3.4 Learned Model Generalization

The model was trained using SIDD and FiveK sRGB datasets with separate task identifiers.

Potential issue:
- Performance may vary on images from unseen smartphone cameras.
- Different lighting and noise distributions may reduce accuracy.
- Generalization to RAW Bayer data has not been demonstrated.

Mitigation:
- Test on additional smartphone datasets.
- Evaluate unseen camera conditions.
- Add RAW data and a dedicated RAW processing pipeline.

### 3.5 Fusion Weight Sensitivity

The output depends on the learned/classical fusion weights.

Preliminary ablation showed that 70/30 outperformed 85/15, 90/10 and 100/0 on one tested image.

Limitation:
- The best fusion weight may vary across scenes.
- One-image results cannot establish a universal optimum.

Future improvement:
- Evaluate fusion weights across more images.
- Explore adaptive fusion based on estimated noise and image content.

## 4. Known Scope Limitations

- Current evaluation uses sRGB images, not RAW Bayer sensor data.
- Real-time smartphone hardware execution has not been demonstrated.
- Quantization and pruning have not yet been evaluated.
- LPIPS has been calculated on resized 512 × 512 images.
- Preliminary metrics are not representative of the full SIDD dataset.

## 5. Conclusion

HybridISP demonstrates that combining classical image processing with a learned ISP model can improve reconstruction quality on the tested SIDD image.

However, full-resolution CPU inference remains computationally expensive. Further evaluation, GPU acceleration, optimization and testing on unseen camera data are required before claiming real-time smartphone deployment.
