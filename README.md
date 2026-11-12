# HybridISP — Intelligent Smartphone Image Signal Processing

**Problem Statement:** P2 — Intelligent and Computationally Efficient Smartphone ISP  
**Team ID:** T90

## 1. Overview

HybridISP is a hybrid image signal processing system that combines
classical computer vision techniques with a task-conditioned deep
learning model.

The system provides three processing approaches:

- Classical ISP: traditional image filtering.
- Learned ISP: deep learning-based image enhancement.
- Hybrid ISP: weighted fusion of classical and learned outputs.

The objective is to improve image quality while evaluating computational
cost, image quality metrics, and model complexity.

## 2. System Architecture

```text
            Input RGB Image
                 |
                 v
          Image Preprocessing
                 |
      +---------------------+
      |                     |
      v                     v
Classical ISP          Learned ISP
(Bilateral Filter)     (HybridISPNet)
      |                     |
      +----------+----------+
                 |
                 v
          Hybrid Fusion
                 |
                 v
          Output RGB Image
                 |
                 v
       Quality Evaluation
3. Technology Stack
- Python
- PyTorch
- Flask
- OpenCV
- NumPy
- Scikit-image
- HTML, CSS, JavaScript
- Docker
4. Project Structure
HybridISP/
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── models/
│   ├── best_model.pth
│   ├── hybrid_isp_model.py
│   └── model_loader.py
├── isp/
│   ├── classical_isp.py
│   ├── learned_isp.py
│   ├── hybrid_isp.py
│   └── metrics.py
├── evaluation/
│   ├── evaluate_sidd.py
│   ├── advanced_metrics.py
│   ├── ablation_study.py
│   ├── benchmark_tiles.py
│   ├── failure_analysis.md
│   └── results/
├── templates/
│   └── index.html
└── static/
    └── css/
        └── style.css


5. Installation and Execution
Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd HybridISP


Create a virtual environment
python -m venv .venv


Activate it on Windows:
.venv\Scripts\activate


Install dependencies
pip install -r requirements.txt


Run the application
python app.py


Open the local address displayed in the terminal.
6. Model Information
- Architecture: HybridISPNet
- Parameters: 3,086,051
- Checkpoint size: approximately 35.4 MB
- Training datasets: SIDD sRGB and MIT-Adobe FiveK
- Training framework: PyTorch
The model was trained using task-conditioned learning for image
denoising and enhancement.
7. Evaluation
The project includes evaluation scripts for:
- PSNR
- SSIM
- Delta E 2000
- LPIPS
- Model parameter count
- Checkpoint size
- Inference latency
- Ablation analysis
- Tile-size benchmarking
The currently committed evaluation results use limited image samples
and must be interpreted as preliminary experiments, not full-dataset
benchmark results.

See evaluation/results/ and evaluation/failure_analysis.md.
8. Current Limitations
- The current datasets provide sRGB images, not RAW Bayer sensor data.
  Therefore, this implementation does not demonstrate a complete
  RAW-to-RGB camera ISP pipeline.
- CPU inference on high-resolution images is computationally expensive.
- Evaluation results are preliminary and require broader testing.
- Performance may vary with image resolution and hardware.
9. Docker Execution
Build the Docker image:
docker build -t hybridisp .


Run the container:
docker run -p 5000:5000 hybridisp


10. Team
Team ID: T90

Problem Statement: P2 — Intelligent and Computationally Efficient Smartphone ISP