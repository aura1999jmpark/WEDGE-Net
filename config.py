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

# Device setting is handled automatically in main scripts (cuda if available),
# but you can force specific device indices here if needed.
SEED = 42

# ==============================================================================
# [3] WEDGE-Net Model Configuration
# ==============================================================================
# Wavelet Kernel Type for the Frequency Stream.
# Options: 'haar' (Hardware-efficient, sharp edges), 'bior2.2' (Smoother boundaries)
WAVELET_TYPE = 'haar'

# Enable/Disable the Semantic Stream (ResNet Global Attention).
# Set True for WEDGE-Net (Full), False for Frequency Stream Only (Ablation).
USE_SEMANTIC = True

# ==============================================================================
# [4] Memory Bank & Sampling Configuration
# ==============================================================================
# Directory where the trained memory banks (.pt files) will be saved.
# If the directory does not exist, it will be automatically created.
# NOTE: If this variable is not specified, a 'WEDGE-Net' folder will be created 
#       by default, and the .pt files will be saved there.
SAVE_DIR = "WEDGE-Net_realC"


# Ratio of features to retain in the memory bank.
# Options:
#   1.0   : 100% (Full Memory)
#   0.1   : 10% (WEDGE-Net Default)
#   0.01  : 1% (Extreme Compression)
#   'all' : Automatically generate and save all three versions (100%, 10%, 1%)
#           in their respective sub-folders.
SAMPLING_RATIO = '0.1'

# Sampling Strategy to use when SAMPLING_RATIO < 1.0.
# Options: 
#   'coreset' : K-Center Greedy (Recommended, preserves manifold boundaries)
#   'random'  : Random subsampling (Faster, but less accurate)
SAMPLING_METHOD = 'coreset'
#SAMPLING_METHOD = 'random'

# ==============================================================================
# [5] Path Configuration for Gap Comparison
# ==============================================================================
# Directory for our main proposed model (Semantic ON)
OurModel_DIR = "WEDGE-Net"

# Directory for the comparison model (Semantic OFF / Reference)
CompareModel_DIR = "patch_core_pt"

# Result CSV Filename (Saved inside OurModel_DIR)
GAP_RESULT_CSV = "result_gap_via_sem_onoff.csv"