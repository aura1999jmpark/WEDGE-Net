# WEDGE-Net: Wavelet-Driven Memory-Efficient Anomaly Detection for Industrial Edge Computing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

# WEDGE-Net

Official PyTorch Implementation of WEDGE-Net

> Abstract: WEDGE-Net is a frequency-aware anomaly detection framework designed for resource-constrained edge devices. By leveraging Discrete Wavelet Transform (DWT) and a Semantic Module, it achieves SOTA-level performance using only 10% of the memory bank, delivering 270 FPS on an RTX 4090.

## 🏗️ Architecture
Figure 1: Overview of WEDGE-Net architecture.
<img width="2848" height="1504" alt="architecture11" src="https://github.com/user-attachments/assets/c7ea825a-0dc3-4314-bffb-500036c661b4" />

## 🚀 Key Features
- Extreme Efficiency: 7.1x faster inference (270 FPS) compared to PatchCore.
- Memory Efficient: Uses only 10% of the memory bank via Greedy Coreset.
- Noise Robust: Filters out environmental noise using DWT (Frequency Stream).

## 🛠️ Installation
```bash
git clone [https://github.com/aura1999jmpark/WEDGE-Net.git](https://github.com/aura1999jmpark/WEDGE-Net.git)
cd WEDGE-Net
pip install -r requirements.txt
