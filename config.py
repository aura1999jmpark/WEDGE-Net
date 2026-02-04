import os

# ==============================================================================
# [1] Data Configuration
# ==============================================================================
# Path to the MVTec AD dataset root directory.
DATA_PATH = './mvtec_ad' 

# Target Category for training/evaluation.
# Options: 'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
#          'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
# Set to 'all' to process all categories sequentially.
CATEGORY = 'tile'

# ==============================================================================
# [2] System & Training Configuration
# ==============================================================================
# Input image size (Resize). Standard PatchCore uses 224 or 256.
IMAGE_SIZE = 224

# Batch size for feature extraction.
BATCH_SIZE = 32

# Number of CPU workers for data loading.
NUM_WORKERS = 4

# Random Seed for reproducibility.
SEED = 42

# ==============================================================================
# [3] WEDGE-Net Model Configuration
# ==============================================================================
# Wavelet Kernel Type for the Frequency Stream.
# Options: 'haar' (Hardware-efficient, sharp edges), 'bior2.2' (Smoother boundaries)
WAVELET_TYPE = 'haar'
#WAVELET_TYPE = 'bior2.2'

# Enable/Disable the Semantic Stream (ResNet Global Attention).
# Set True for WEDGE-Net (Full Proposed Method).
# Set False for Frequency Stream Only (Ablation Baseline).
USE_SEMANTIC = True
#USE_SEMANTIC = False

# ==============================================================================
# [4] Memory Bank & Sampling Configuration
# ==============================================================================
# Directory where the trained memory banks (.pt files) will be saved.
# NOTE: Main training scripts use this variable to save models.
SAVE_DIR = "WEDGE-Net"

# Ratio of features to retain in the memory bank.
# Options:
#   1.0   : 100% (Full Memory)
#   0.1   : 10% (WEDGE-Net Default)
#   0.01  : 1% (Extreme Compression)
#   'all' : Automatically generate and save all three versions.
SAMPLING_RATIO = '0.1'

# Sampling Strategy to use when SAMPLING_RATIO < 1.0.
# Options: 
#   'coreset' : K-Center Greedy (Recommended, preserves manifold boundaries)
#   'random'  : Random subsampling (Faster, but less accurate)
SAMPLING_METHOD = 'coreset'
# SAMPLING_METHOD = 'random'

# ==============================================================================
# [5] Path Configuration for Evaluation & Visualization
# ==============================================================================
# -------------------------------------------------------
# A. Directory Definitions (Physical Paths)
# -------------------------------------------------------
# 1. Main Proposed Model (Semantic ON)
#    - Path to the model trained with USE_SEMANTIC=True
#    - [NOTE] Change '10pct' to '1pct' or '100pct' if you want to evaluate other ratios.
SemanticON_DIR = "WEDGE-Net/10pct"

# 2. Internal Ablation Baseline (Semantic OFF)
#    - Path to the model trained with USE_SEMANTIC=False
#    - Used ONLY for Table 7 (Score Gap Analysis)
#    - [NOTE] Ensure the ratio matches SemanticON_DIR (e.g., if testing 1pct, point this to the 1pct OFF model).
SemanticOFF_DIR = "WEDGE-Net_Sem_off/10pct"

# 3. External SOTA Comparison (PatchCore)
#    - Used for Figure 5 & 6 (Noise/Color Robustness)
#    - [IMPORTANT] Please specify the directory containing the PatchCore .pt files.
PatchCore_DIR = "patch_core_pt"

# -------------------------------------------------------
# B. Backward Compatibility (Aliasing)
#    * Do not modify this section unless you change the scripts.
# -------------------------------------------------------

# 'OurModel_DIR' is used by visualization scripts (Figures) and Gap Analysis.
OurModel_DIR = SemanticON_DIR

# 'CompareModel_DIR' is used by Noise/Color Robustness scripts (Figures 5, 6).
# For these figures, we compare against PatchCore (SOTA).
CompareModel_DIR = PatchCore_DIR

# ==============================================================================
# [6] Result Output Configuration
# ==============================================================================
# Result CSV Filename for Ablation Study (Saved inside SemanticON_DIR)
GAP_RESULT_CSV = "result_gap_via_sem_onoff.csv"
