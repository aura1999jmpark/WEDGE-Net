# WEDGE-Net: Wavelet-Driven Memory-Efficient Anomaly Detection for Industrial Edge Computing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

# WEDGE-Net

Official PyTorch Implementation of WEDGE-Net

> Abstract: WEDGE-Net is a frequency-aware anomaly detection framework designed for resource-constrained edge devices. By leveraging Discrete Wavelet Transform (DWT) and a Semantic Module, it achieves SOTA-level performance using only 10% of the memory bank, delivering 270 FPS on an RTX 4090.

## Architecture
Figure 1: Overview of WEDGE-Net architecture.
<img width="2848" height="1504" alt="architecture11" src="https://github.com/user-attachments/assets/c7ea825a-0dc3-4314-bffb-500036c661b4" />

## Key Features
- Extreme Efficiency: 7.1x faster inference (270 FPS) compared to PatchCore.
- Memory Efficient: Uses only 10% of the memory bank via Greedy Coreset.
- Noise Robust: Filters out environmental noise using DWT (Frequency Stream).

## Dependencies
- Python 3.8+
- PyTorch 1.10+
- Torchvision
- Scipy, Scikit-learn, Tqdm, Matplotlib

- ## ⚙️ Configuration (`config.py`)

You can control all experimental settings in `config.py`. Key parameters are explained below:

### 1. **Dataset & Category**
| Parameter | Description |
| :--- | :--- |
| `DATA_PATH` | Path to the MVTec AD dataset root (default: `./mvtec_ad`) |
| `CATEGORY` | Target category to process. <br> - **Single:** `'bottle'`, `'tile'`, etc. <br> - **All:** `'all'` (processes all 15 categories sequentially) |

### 2. Model Settings
| Parameter | Description |
| :--- | :--- |
| `WAVELET_TYPE` | Wavelet kernel type (`'haar'` or `'bior2.2'`). |
| `USE_SEMANTIC` | Toggle the Semantic Attention Stream (`True` / `False`). |

### 3. Sampling & Memory Bank
| Parameter | Description |
| :--- | :--- |
| `SAVE_DIR` | Directory where the trained memory banks (`.pt`) will be saved. |
| `SAMPLING_RATIO` | Coreset sampling ratio. <br> - `1.0`: 100% (Full Memory) <br> - `0.1`: 10% (Proposed) <br> - `'all'`: Generates 100%, 10%, and 1% versions simultaneously. |
| `SAMPLING_METHOD` | `'coreset'` (Recommended) or `'random'`. |

---

## Usage Guide

### 1. Training (Feature Extraction & Coreset Sampling)
This script extracts features from normal training images and saves the memory bank (`.pt` file).

How to run:
```bash
python train.py

## Installation
```bash
git clone [https://github.com/aura1999jmpark/WEDGE-Net.git](https://github.com/aura1999jmpark/WEDGE-Net.git)
cd WEDGE-Net
pip install -r requirements.txt


