

1RUA25CSE7019,1RUA25CSE7008,1RUA25CSE7011,1RUA25CSE7010,1RUA25CSE7005



\# Comparative Evaluation Report — HybridISP



\*\*Problem Statement:\*\* P2 — Intelligent and Computationally Efficient Smartphone ISP  

\*\*Team ID:\*\* T90  

\*\*Project:\*\* HybridISP — Intelligent Smartphone Image Signal Processing



\## 1. Objective



The objective of this evaluation is to compare three image processing approaches:



1\. Classical ISP using bilateral filtering.

2\. Learned ISP using HybridISPNet.

3\. Hybrid ISP combining classical and learned outputs.



The evaluation considers image reconstruction quality, perceptual quality, model complexity, and inference latency.



\## 2. Experimental Setup



\### 2.1 Datasets



The model was trained using:



\- SIDD sRGB dataset for smartphone image denoising.

\- MIT-Adobe FiveK dataset for image enhancement.



The evaluation scripts support paired SIDD noisy and ground-truth images.



\*\*Important dataset limitation:\*\* The available datasets used in this implementation contain sRGB images. RAW Bayer sensor data was not available for this experiment.



\### 2.2 Model Configuration



| Parameter | Value |

|---|---:|

| Model | HybridISPNet |

| Parameters | 3,086,051 |

| Checkpoint size | 35.407 MB |

| Tile size | 256 × 256 |

| Tile overlap | 32 pixels |

| Inference batch size | 1 |

| Hybrid learned weight | 0.85 |

| Hybrid classical weight | 0.15 |



\### 2.3 Hardware and Software



\- Training: Google Colab NVIDIA T4 GPU.

\- Local inference: CPU.

\- Framework: PyTorch.

\- Image processing: OpenCV.

\- Evaluation: NumPy, Scikit-image, LPIPS.



\## 3. Evaluation Metrics



| Metric | Purpose | Preferred Direction |

|---|---|---|

| PSNR | Pixel-level reconstruction quality | Higher |

| SSIM | Structural similarity | Higher |

| Delta E 2000 | Perceptual color difference | Lower |

| LPIPS | Deep perceptual difference | Lower |

| Latency | Processing time | Lower |

| Model size | Storage requirement | Lower |



\## 4. Comparative Results



The following results are from one evaluated SIDD image pair.



\*\*These are preliminary single-image results, not full-dataset benchmark results.\*\*



| Metric | Noisy Input | Classical ISP | Learned ISP | Hybrid ISP |

|---|---:|---:|---:|---:|

| PSNR (dB) | 31.3686 | 38.1520 | 38.6487 | 39.3024 |

| SSIM | 0.6387 | 0.9176 | 0.9125 | 0.9256 |

| Delta E 2000 | 5.2525 | 2.7813 | 2.4569 | 2.2475 |

| LPIPS | 0.0081 | 0.0073 | 0.0082 | 0.0071 |



\### 4.1 Observations



\- Classical filtering substantially improved the noisy input.

\- Learned processing achieved higher PSNR than classical filtering.

\- The learned approach did not outperform classical filtering in SSIM on this sample.

\- Hybrid processing achieved the highest PSNR and SSIM among the evaluated approaches.

\- Hybrid processing also achieved the lowest Delta E and LPIPS in this experiment.



These observations apply only to the evaluated image and cannot establish general performance across the dataset.



\## 5. Computational Performance



One local CPU evaluation recorded the following inference times:



| Processing Method | Latency |

|---|---:|

| Classical ISP | 1.153 seconds |

| Learned ISP | 76.865 seconds |

| Hybrid ISP | 81.079 seconds |



The measurements are hardware- and execution-dependent.



\### Observations



1\. Classical filtering is substantially faster than neural inference.

2\. Learned inference is computationally expensive on the CPU.

3\. Hybrid processing has additional overhead because it executes both processing branches.

4\. The current implementation is not suitable for claiming real-time full-resolution processing on CPU.



\## 6. Model Complexity



| Property | Result |

|---|---:|

| Total parameters | 3,086,051 |

| Checkpoint size | 35.407 MB |

| Inference approach | Tiled inference |

| Tile size | 256 × 256 |

| Overlap | 32 pixels |

| Batch size | 1 |



No verified quantization or pruning experiment has been completed. Therefore, no compression or speedup claim is made.



\## 7. Ablation Study — Hybrid Fusion Weights



The fusion equation is:



\\\[

I\_{hybrid}=\\alpha I\_{learned}+(1-\\alpha)I\_{classical}

\\]



The following results were obtained from one evaluated image pair.



| Learned Weight | Classical Weight | PSNR (dB) | SSIM | Delta E | LPIPS |

|---:|---:|---:|---:|---:|---:|

| 0.70 | 0.30 | 39.7487 | 0.9345 | 2.1233 | 0.0064 |

| 0.85 | 0.15 | 39.3024 | 0.9256 | 2.2475 | 0.0071 |

| 0.90 | 0.10 | 39.1031 | 0.9217 | 2.3088 | 0.0075 |

| 1.00 | 0.00 | 38.6487 | 0.9125 | 2.4569 | 0.0082 |



\### Ablation Observations



\- The 70/30 fusion ratio produced the best scores across the four reported metrics on this sample.

\- Increasing the classical contribution improved the measured result in this particular experiment.

\- The 100/0 configuration represents learned-only processing.

\- The current application default remains 85/15.

\- The 70/30 result is preliminary and does not establish a globally optimal fusion ratio.



\## 8. Tile Configuration Experiments



A local CPU benchmark on one high-resolution image produced these observations:



| Tile Size | Overlap | Batch Size | Observed Time |

|---:|---:|---:|---:|

| 128 | 32 | 1 | 73.61 seconds |

| 256 | 32 | 1 | 67.63 seconds |

| 256 | 32 | 2 | 77.13 seconds |

| 512 | 32 | 1 | 91.43 seconds |



The 256 × 256 tile with batch size 1 was the fastest among these tested configurations.



This is a limited hardware-specific experiment, not proof that this configuration is optimal on other devices.



\## 9. Failure Analysis and Limitations



\### 9.1 CPU Inference Latency



The learned model takes a significant amount of time to process high-resolution images on CPU.



\*\*Impact:\*\* Limits interactive and real-time usage.



\### 9.2 Limited Evaluation Samples



The detailed metric comparison and fusion ablation use only one image pair.



\*\*Impact:\*\* Results may not generalize to other scenes, lighting conditions, or noise levels.



\### 9.3 Dataset Limitations



The available datasets contain sRGB images rather than RAW Bayer sensor data.



\*\*Impact:\*\* A complete RAW-to-RGB smartphone ISP pipeline has not been demonstrated.



\### 9.4 Fusion Sensitivity



The measured output quality changes with the selected fusion weight.



\*\*Impact:\*\* The default fusion ratio requires broader validation.



\### 9.5 Model Training Limitations



The model was trained for a limited number of epochs.



\*\*Impact:\*\* Further training and validation may improve generalization and restoration quality.



\### 9.6 Compression and Mobile Deployment



Quantization, pruning, and mobile hardware acceleration have not been validated.



\*\*Impact:\*\* Computational efficiency on actual smartphone hardware remains unproven.



\## 10. Conclusion



The preliminary experiment demonstrates that combining classical filtering with learned image processing can improve the measured image quality for the evaluated SIDD sample.



The hybrid approach achieved a PSNR of 39.3024 dB and SSIM of 0.9256, compared with 38.1520 dB and 0.9176 for classical processing.



However, the learned pipeline introduces significant CPU inference latency. The current system therefore demonstrates a quality-oriented hybrid processing prototype rather than a production-ready real-time smartphone ISP.



Broader dataset evaluation, RAW sensor support, model compression, and hardware optimization remain important future improvements.



\## 11. Reproducibility



Evaluation scripts are available in the `evaluation/` directory.



Key files:



\- `evaluation/evaluate\_sidd.py`

\- `evaluation/advanced\_metrics.py`

\- `evaluation/ablation\_study.py`

\- `evaluation/benchmark\_tiles.py`

\- `evaluation/failure\_analysis.md`



Raw evaluation outputs are stored in `evaluation/results/`.



