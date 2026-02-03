# WEDGE-Net: Wavelet-Driven Memory-Efficient Anomaly Detection for Industrial Edge Computing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

**Official PyTorch Implementation of WEDGE-Net**

> **Abstract:** > WEDGE-Net is a frequency-aware anomaly detection framework designed for resource-constrained edge devices. By leveraging Discrete Wavelet Transform (DWT) and a Semantic Module, it achieves SOTA-level performance using only **10% of the memory bank**, delivering **270 FPS** on an RTX 4090.

## 🏗️ Architecture
![Architecture](./assets/architecture_last9.jpg)
*Figure 1: Overview of WEDGE-Net architecture.*

## 🚀 Key Features
- **Extreme Efficiency:** 7.1x faster inference (270 FPS) compared to PatchCore.
- **Memory Efficient:** Uses only **10%** of the memory bank via Greedy Coreset.
- **Noise Robust:** Filters out environmental noise using DWT (Frequency Stream).

## 🛠️ Installation
```bash
git clone [https://github.com/aura1999jmpark/WEDGE-Net.git](https://github.com/aura1999jmpark/WEDGE-Net.git)
cd WEDGE-Net
pip install -r requirements.txt# WEDGE-Net
