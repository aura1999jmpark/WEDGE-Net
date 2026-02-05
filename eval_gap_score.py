import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
import os
import numpy as np
import csv
import sys
import contextlib  # [Added] Required to temporarily suppress stdout
from model import WEDGE_Net
from dataset import MVTecDataset
import config  # Import updated config

# ==============================================================================
# 1. Settings & Configuration
# ==============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Paths from Config
DIR_OFF = config.SemanticOFF_DIR  # Baseline (Semantic OFF)
DIR_ON  = config.SemanticON_DIR   # Proposed (Semantic ON)

# Safety Check: Ensure the user set the OFF directory
if not DIR_OFF or DIR_OFF == "":
    print("\n" + "!"*60)
    print(" [Error] 'SemanticOFF_DIR' is empty in config.py!")
    print(" Please specify the path to the model trained with 'use_semantic=False'.")
    print(" It is required to reproduce 'Anomaly Score Margin' in the Discussion section")
    print("!"*60 + "\n")
    sys.exit(1)

# CSV Output Path
CSV_NAME = config.GAP_RESULT_CSV
CSV_PATH = os.path.join(DIR_ON, CSV_NAME)

# Category Definitions
TEXTURE_CLASSES = sorted(['carpet', 'grid', 'leather', 'tile', 'wood'])
OBJECT_CLASSES = sorted([
    'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 
    'pill', 'screw', 'toothbrush', 'transistor', 'zipper'
])
ALL_CATEGORIES = TEXTURE_CLASSES + OBJECT_CLASSES

# Determine Target Ratio based on Config
# Converts config value to string to handle both float (0.01) and string ('0.01')
target_ratio_str = str(config.SAMPLING_RATIO)

if '0.01' in target_ratio_str:
    TARGET_RATIO = "1pct"
elif '0.1' in target_ratio_str:
    TARGET_RATIO = "10pct"
elif '1.0' in target_ratio_str or '1' == target_ratio_str:
    TARGET_RATIO = "100pct"
else:
    # Fallback default
    TARGET_RATIO = "10pct"
    print(f"⚠️ Warning: Config ratio '{config.SAMPLING_RATIO}' not recognized. Defaulting to '10pct'.")

print(f"🎯 Target Ratio Set: {TARGET_RATIO} (from config: {config.SAMPLING_RATIO})")


# ==============================================================================
# 2. Utility Functions
# ==============================================================================

def extract_features_1536(model, x):
    """
    Feature Extractor for Semantic OFF Model.
    Forces extraction of 1536-dim features to match the baseline architecture.
    """
    features = []
    def hook(module, input, output):
        features.append(output)
        
    # Hook registration based on architecture
    if hasattr(model, 'encoder'):
        h1 = model.encoder.layer2.register_forward_hook(hook)
        h2 = model.encoder.layer3.register_forward_hook(hook)
    else:
        h1 = model.layer2.register_forward_hook(hook)
        h2 = model.layer3.register_forward_hook(hook)
        
    _ = model(x)
    h1.remove(); h2.remove()
    
    # Standard Preprocessing (AvgPool + Interpolate)
    f1 = F.avg_pool2d(features[0], 3, 1, 1)
    f2 = F.avg_pool2d(features[1], 3, 1, 1)
    f2 = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=False)
    
    return torch.cat([f1, f2], dim=1)

def get_model_path(base_dir, category):
    """
    Smart Path Finder for .pt files.
    Searches for model checkpoints in prioritized order.
    """
    # Priority 1: Sub-directory (e.g., base/10pct/model_bottle_10pct.pt)
    path1 = os.path.join(base_dir, TARGET_RATIO, f"model_data_{category}_{TARGET_RATIO}.pt")
    if os.path.exists(path1): return path1
    
    # Priority 2: Root directory with suffix (e.g., base/model_bottle_10pct.pt)
    path2 = os.path.join(base_dir, f"model_data_{category}_{TARGET_RATIO}.pt")
    if os.path.exists(path2): return path2

    # Priority 3: Legacy naming (e.g., base/model_bottle.pt)
    path3 = os.path.join(base_dir, f"model_data_{category}.pt")
    if os.path.exists(path3): return path3
    
    return None

def calculate_gap(category, mode):
    """
    Calculates the Anomaly Score Gap between Defect and Normal samples.
    """
    # [Modified] Suppress initialization logs (e.g., "Initialized. Wavelet: HAAR")
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        if mode == 'OFF':
            model = WEDGE_Net(use_semantic=False).to(DEVICE)
            base_dir = DIR_OFF
        else:
            model = WEDGE_Net(use_semantic=True).to(DEVICE)
            base_dir = DIR_ON
    
    pt_path = get_model_path(base_dir, category)
    if pt_path is None:
        return None

    # [Log] Print the verified model path
    print(f"    [{mode}] Found at: {pt_path}")

    # Load Checkpoint
    try:
        checkpoint = torch.load(pt_path, map_location=DEVICE)
        bank = checkpoint['memory_bank'] if isinstance(checkpoint, dict) else checkpoint
        model.eval()
    except Exception as e:
        print(f"Error loading {pt_path}: {e}")
        return None

    # Load Test Data
    ds = MVTecDataset(root_dir=config.DATA_PATH, category=category, phase='test')
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    normal_scores = []
    defect_scores = []

    with torch.no_grad():
        for img, label, _, _ in loader:
            img = img.to(DEVICE)
            
            # Feature Extraction
            if mode == 'OFF':
                features = extract_features_1536(model, img)
            else:
                features, _ = model(img)
            
            # Anomaly Scoring (Mahalanobis-like distance)
            b, c, h, w = features.shape
            features_flat = features.view(c, -1).permute(1, 0)
            
            dists = torch.cdist(features_flat, bank)
            s_map = dists.min(dim=1)[0].reshape(h, w).cpu().numpy()
            s_map = gaussian_filter(s_map, sigma=4)
            score = s_map.max()

            if label.item() == 0:
                normal_scores.append(score)
            else:
                defect_scores.append(score)

    if not normal_scores or not defect_scores:
        return 0.0
        
    # Return Score Gap (Defect - Normal)
    return np.mean(defect_scores) - np.mean(normal_scores)

# ==============================================================================
# 3. Main Execution Flow
# ==============================================================================
def main():
    print(f" [Discussion] Score Gap Analysis")
    print(f" - Target Ratio  : {TARGET_RATIO}")
    print(f" - Semantic OFF Dir: {DIR_OFF}")
    print(f" - Semantic ON  Dir: {DIR_ON}")
    print("-" * 60)
    
    results = []
    tex_gaps_off, tex_gaps_on = [], []
    obj_gaps_off, obj_gaps_on = [], []

    for cat in ALL_CATEGORIES:
        group_type = "Texture" if cat in TEXTURE_CLASSES else "Object"
        
        # Explicit newline to separate category logs
        print(f" >> [Processing] Category: {cat.upper()} ({group_type})")
        
        # Calculate Gaps
        gap_off = calculate_gap(cat, 'OFF')
        gap_on = calculate_gap(cat, 'ON')
        
        if gap_off is not None and gap_on is not None:
            # Calculate Improvement (%)
            if abs(gap_off) < 1e-9: imp = 0.0
            else: imp = ((gap_on - gap_off) / abs(gap_off)) * 100
            
            results.append({
                'Category': cat, 'Type': group_type,
                'Gap_OFF': round(gap_off, 5), 'Gap_ON': round(gap_on, 5),
                'Improvement(%)': round(imp, 2)
            })
            
            if group_type == "Texture":
                tex_gaps_off.append(gap_off); tex_gaps_on.append(gap_on)
            else:
                obj_gaps_off.append(gap_off); obj_gaps_on.append(gap_on)
        else:
            print(f"    (Skipped: Model checkpoint not found)")

    print("\nAnalysis Complete.")

    # Calculate Averages
    avg_rows = []
    if tex_gaps_off:
        m_off, m_on = np.mean(tex_gaps_off), np.mean(tex_gaps_on)
        m_imp = ((m_on - m_off) / abs(m_off)) * 100
        avg_rows.append({'Category': 'AVG (Texture)', 'Type': 'Texture', 
                         'Gap_OFF': round(m_off, 5), 'Gap_ON': round(m_on, 5), 
                         'Improvement(%)': round(m_imp, 2)})
        
    if obj_gaps_off:
        m_off, m_on = np.mean(obj_gaps_off), np.mean(obj_gaps_on)
        m_imp = ((m_on - m_off) / abs(m_off)) * 100
        avg_rows.append({'Category': 'AVG (Object)', 'Type': 'Object', 
                         'Gap_OFF': round(m_off, 5), 'Gap_ON': round(m_on, 5), 
                         'Improvement(%)': round(m_imp, 2)})

    # Save to CSV
    if not os.path.exists(DIR_ON): os.makedirs(DIR_ON)
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Category', 'Type', 'Gap_OFF', 'Gap_ON', 'Improvement(%)'])
        writer.writeheader()
        writer.writerows(results)
        writer.writerow({}) # Empty row for separation
        writer.writerows(avg_rows)

    print(f"Saved to: {CSV_PATH}")
    for row in avg_rows:
        print(f" >> {row['Category']}: {row['Gap_OFF']} -> {row['Gap_ON']} ({row['Improvement(%)']}%)")

if __name__ == "__main__":
    main()
