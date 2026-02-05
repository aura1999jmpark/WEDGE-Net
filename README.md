# WEDGE-Net: Wavelet-Driven Memory-Efficient Anomaly Detection for Industrial Edge Computing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

# WEDGE-Net

Official PyTorch Implementation of WEDGE-Net

> Abstract: WEDGE-Net is a frequency-aware anomaly detection framework designed for resource-constrained edge devices. By leveraging Discrete Wavelet Transform (DWT) and a Semantic Module, it achieves SOTA-level performance using only 10% of the memory bank, delivering 270 FPS on an RTX 4090.

## Architecture
Figure 1: Overview of WEDGE-Net architecture.
<img width="2848" height="1504" alt="architecture11" src="https://github.com/user-attachments/assets/c7ea825a-0dc3-4314-bffb-500036c661b4" />

## ✨ Key Features
- **Extreme Efficiency:** 7.1x faster inference (**270 FPS**) compared to PatchCore.
- **Memory Efficient:** Uses only **10%** of the memory bank via Greedy Coreset.
- **Noise Robust:** Filters out environmental noise using DWT (Frequency Stream).
- **Plug-and-Play:** Simple architecture compatible with standard ResNet backbones.

## Dependencies
- Python 3.9
- PyTorch 1.10+
- Torchvision
- Scipy, Scikit-learn, Tqdm, Matplotlib

---

## 🛠️ Installation
```bash
git clone [https://github.com/aura1999jmpark/WEDGE-Net.git](https://github.com/aura1999jmpark/WEDGE-Net.git)
cd WEDGE-Net
pip install -r requirements.txt
```
---

## ⚙️ Configuration (`config.py`)

You can control all experimental settings in `config.py`. Key parameters are explained below:

### 1. Dataset & Category
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
| `SAVE_DIR` | **Root Directory** where the memory banks will be saved (e.g., `"WEDGE-Net"`). |
| `SAMPLING_RATIO` | Coreset sampling ratio. <br> **Note:** The script **automatically creates sub-folders** based on this value: <br> - `'all'` $\rightarrow$ Generates `100pct/`, `10pct/`, and `1pct/` folders simultaneously. <br> - `0.1` $\rightarrow$ Generates a `10pct/` folder. <br> - `0.01` $\rightarrow$ Generates a `1pct/` folder. |
| `SAMPLING_METHOD` | `'coreset'` (Recommended) or `'random'`. |

> **[IMPORTANT] Dynamic Path Loading for Visualization**
> All visualization and evaluation scripts (e.g., `visualize_noise_robustness.py`, `eval_color_robustness.py`) automatically detect the model file based on the `SAMPLING_RATIO` set in `config.py`.
>
> *Example:* If you set `SAMPLING_RATIO = '0.1'` and `CATEGORY = 'tile'`, the scripts will automatically attempt to load:
> ```text
> [SAVE_DIR]/10pct/model_data_tile_10pct.pt
> ```
> *Please ensure that the model corresponding to the config ratio exists before running the scripts.*

### 4. Path Configuration (Comparison & Ablation)
To reproduce the specific figures and tables in the paper, you need to specify the model paths in `config.py`:

| Parameter | Description |
| :--- | :--- |
| `SemanticON_DIR` | Path to the main proposed model (WEDGE-Net). <br> *Used in: Visualization, Gap Analysis* |
| `SemanticOFF_DIR` | Path to the baseline model trained with `USE_SEMANTIC=False`. <br> *Used in: Table 7 (Analysis of Anomaly Score Margins)* |
| `PatchCore_DIR` | Path to pre-trained PatchCore models (.pt files). <br> *Used in: Figure 5 & 6 (Robustness Experiments)* |

---

## 🚀 Usage Guide

### 1. Training (Feature Extraction & Memory Bank)
Extracts features from training images and constructs the memory bank (`.pt` file). All settings are controlled via `config.py`.

**Key Configurations to Check:**
- `CATEGORY`: Target class (e.g., `'tile'` or `'all'`).
- `SAMPLING_RATIO`: Memory size to retain (e.g., `0.1` for 10%, `0.01` for 1%, or `'all'`).
- `SAMPLING_METHOD`: Sampling strategy (`'coreset'` or `'random'`).
```bash
python train.py
```
> **Note:**
>
> - Multi-Ratio Mode: If `SAMPLING_RATIO` is set to `'all'`, the script automatically generates sub-folders for **100pct**, **10pct**, and **1pct** under your `SAVE_DIR`.
> - Multi-Category Mode: If `CATEGORY` is set to 'all', the script sequentially processes **all 15 MVTec AD categories** (from Bottle to Zipper) and saves the trained models into the sub-folder corresponding to your `SAMPLING_RATIO`.

### 2. Evaluation (Performance Metric)
Calculate the Image-level AUROC to verify the model's detection performance.
```bash
# To calculate AUROC for the trained model
python evaluation.py
```
> **Output:** It will print the AUROC score (e.g., `[tile] Image-level AUROC: 0.9912`) based on the current configuration.

### 3. Visualization (Qualitative Analysis)
Generate 6-column qualitative result figures, including Frequency and Semantic Attention Maps (corresponding to Figure 3 in the paper).
```bash
# To generate visualization results
python test.py
```
> **Results:** Output images are saved in `[SAVE_DIR]/[RATIO]/results/[CATEGORY]/`. These figures highlight how the **Proposed Baseline** captures high-frequency structures and semantic context.
) will be generated, documenting the performance improvements.

### 4. Performance

WEDGE-Net demonstrates **state-of-the-art inference speed**, suitable for real-time industrial edge applications. By significantly reducing the memory bank size while maintaining high accuracy, it achieves up to **18.4x speedup** compared to the full-memory baseline.

### ⚡ Inference Speed & Accuracy Comparison
Experiments were conducted on the **MVTec AD** dataset using an NVIDIA RTX 4090.

| Model | Memory Bank | AUROC (Avg) | FPS (Inference) | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| PatchCore (Ref) | 100% | 99.2% | 37 | 1.0x |
| **WEDGE-Net (Ours)** | 10% | 99.1% | **265** | **7.1x** |
| **WEDGE-Net (Ours)** | **1%** | **98.5%** | **687** | **18.4x** |
**How to run:**
1. Set the target ratio in `config.py` (e.g., `SAMPLING_RATIO = 0.01`).
2. Run the benchmark script:
   ```bash
   python benchmark_fps.py
   ```
> **Note:**
> * **FPS** values are averaged across all 15 MVTec AD categories (batch size = 1).
> * **1% Coreset** setting delivers **680+ FPS**, enabling ultra-low latency processing on edge devices.
---

## 🛡️ Robustness Experiments
We provide evaluation scripts to reproduce the robustness experiments (Figure 6) demonstrated in the paper. These scripts measure AUROC scores under varying environmental distortions and automatically generate result graphs (`.png`) and data tables (`.csv`).

### 1. Gaussian Noise Robustness
Evaluates the model's stability against image noise ($\sigma = 0 \sim 40$).

```bash
python eval_noise_robustness.py
```
> **Output:** full_noise_comparison_smoothed_{CATEGORY}_{RATIO}.csv: AUROC scores for each noise level. figure5_noise_robustness_{CATEGORY}_{RATIO}.png: The robustness curve graph.(matches Figure 5 in the paper).
>
> **Note:** Comparison with Baseline: If a pre-trained PatchCore checkpoint exists in the patch_core_pt/ directory, the scripts will include it in the comparison. If not, they will gracefully run the evaluation for WEDGE-Net only.


### 2. Color Jitter Robustness
Evaluates the model's stability against domain shifts (Brightness, Contrast, Saturation, Hue).
```bash
python eval_color_robustness.py
```
> **Output:** final_color_comparison_{CATEGORY}_{RATIO}.csv: AUROC scores for each intensity factor (0.0 ~ 3.0). figure6_color_robustness_{CATEGORY}_{RATIO}.png: The robustness curve graph (matches Figure 6 in the paper).
> 
> **Note:** Comparison with Baseline: If a pre-trained PatchCore checkpoint exists in the patch_core_pt/ directory, the scripts will include it in the comparison. If not, they will gracefully run the evaluation for WEDGE-Net only.

### 3. Qualitative Analysis: Gaussian Noise (Visualization)
Visualizes the anomaly maps of normal samples under increasing Gaussian noise levels to demonstrate stability.
```bash
python visualize_noise_robustness.py
```
> **Output:** Figure_Noise_Robustness_{CATEGORY}.png (4x4 Visualization Tile)

### 4. Qualitative Analysis: Color Jitter (Visualization)  
Visualizes the anomaly maps under domain shifts (Brightness, Contrast, Saturation) to verify that the model does not misclassify altered normal images.
```bash
python visualize_color_robustness.py
```
> **Output:** Figure_Color_Robustness_{CATEGORY}_Final.png (4x4 Visualization Tile)
>
> **Note:**
>
> - **Dynamic Loading:** The scripts automatically load the WEDGE-Net model corresponding to the `SAMPLING_RATIO` set in `config.py`.
> - **PatchCore Handling:** If the PatchCore checkpoint is missing, the script will **not crash**. Instead, the PatchCore column in the output figure will be displayed as **"Skipped (Gray)"** or blank, allowing you to verify WEDGE-Net's results independently.
---
## Discussion: Score Gap Analysis

Script to reproduce **Table 7 (Analysis of Anomaly Score Margin)** from the Discussion section. This analysis verifies how the Semantic Module improves the separation between normal and defect distributions.

```bash
python eval_gap_score.py
```
> [!IMPORTANT]
> **Prerequisite: You must set SemanticOFF_DIR in config.py to point to a model trained with USE_SEMANTIC=False.**
> 
> **Output:**
> - Console table showing Gap (OFF) vs Gap (ON) and Improvement %.
> - result_gap_via_sem_onoff.csv saved in the directory specified by SemanticON_DIR in config.py.
---
## 📂 Directory Structure

After setting up and running experiments, your directory structure should look like this.
> **Note:**
> 1. The default `SAVE_DIR` is set to `WEDGE-Net`.
> 2. **Sub-folders (e.g., `10pct/`, `1pct/`) are automatically created** inside `SAVE_DIR` based on your `SAMPLING_RATIO` setting, and the trained `.pt` files are saved there.
```text
WEDGE-Net/
├── config.py                 # Main Configuration
├── train.py                  # Training Script
├── evaluation.py             # Evaluation Script
├── test.py                   # Visualization Script
├── mvtec_ad/                 # [Dataset] MVTec AD Root
│
├── patch_core_pt/            # [Comparison] External SOTA Models
│   │                         # * To reproduce Fig 5 & 6 comparison:
│   │                         #   Train PatchCore (WideResNet-50) using the official repo
│   │                         #   and place the .pt files here.
│   └── model_data_tile.pt    # (Optional)
│
├── WEDGE-Net/                # [Default SAVE_DIR] Proposed Model (Semantic ON)
│   ├── 100pct/               # Full Memory Bank
│   ├── 10pct/                # Proposed Baseline (10% Coreset)
│   │   ├── model_data_tile_10pct.pt
│   │   └── results/          # Visualization Figures (Fig 3)
│   └── 1pct/                 # Extreme Compression
│
└── WEDGE-Net_Sem_OFF/# [Discussion] Semantic OFF Model
    │                         # * Used for Table 7 (Score Gap Analysis).
    └── 10pct/                # * Created by changing settings in config.py (see below).
        └── model_data_tile_10pct.pt
```
> [!IMPORTANT]
> **How to generate the WEDGE-Net_Sem_OFF folder:**
> To reproduce the Discussion (Table 7), you must train a separate model with the Semantic Module disabled.
> 1. Open `config.py`.
> 2. Change `USE_SEMANTIC = False`.
> 3. Change `SAVE_DIR = "WEDGE-Net_Sem_OFF"` (or any name you prefer).
> 4. Run `python train.py`.
