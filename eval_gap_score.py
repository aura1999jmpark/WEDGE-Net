import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
import os
import numpy as np
import csv
import sys
import contextlib  # Required to temporarily suppress stdout
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

# Safety Check
if not DIR_OFF or DIR_OFF == "":
    print("\n" + "!"*60)
    print(" [Error] 'SemanticOFF_DIR' is empty in config.py!")
    sys.exit(1)

# CSV Output Path
CSV_NAME = getattr(config, 'GAP_RESULT_CSV', 'result_gap_via_sem_onoff.csv')
CSV_PATH = os.path.join(DIR_ON, CSV_NAME)

# Category Definitions
TEXTURE_CLASSES = sorted(['carpet', 'grid', 'leather', 'tile', 'wood'])
OBJECT_CLASSES = sorted([
    'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 
    'pill', 'screw', 'toothbrush', 'transistor', 'zipper'
])
ALL_CATEGORIES = TEXTURE_CLASSES + OBJECT_CLASSES

# ------------------------------------------------------------------------------
# Ratio Parsing
# ------------------------------------------------------------------------------
def get_ratio_folder_name(ratio_input):
    raw = str(ratio_input).lower()
    if raw == 'all': return "1pct"
    try:
        val = float(raw)
        if abs(val - 0.001) < 1e-6: return "0_1pct"
        elif abs(val - 0.01) < 1e-6: return "1pct"
        elif abs(val - 0.1) < 1e-6: return "10pct"
        elif abs(val - 1.0) < 1e-6: return "100pct"
        else: return f"{int(val * 100)}pct"
    except ValueError: return "1pct"

TARGET_RATIO = get_ratio_folder_name(config.SAMPLING_RATIO)
print(f"🎯 Target Ratio: {config.SAMPLING_RATIO} -> Folder: {TARGET_RATIO}")

# ==============================================================================
# 2. Utility Functions
# ==============================================================================
def get_model_path(base_dir, category):
    """Smart Path Finder for .pt files."""
    # Priority 1: Sub-directory
    path1 = os.path.join(base_dir, TARGET_RATIO, f"model_data_{category}_{TARGET_RATIO}.pt")
    if os.path.exists(path1): return path1
    # Priority 2: Root directory with suffix
    path2 = os.path.join(base_dir, f"model_data_{category}_{TARGET_RATIO}.pt")
    if os.path.exists(path2): return path2
    # Priority 3: Legacy naming
    path3 = os.path.join(base_dir, f"model_data_{category}.pt")
    if os.path.exists(path3): return path3
    return None

def calculate_gap(category, mode):
    """Calculates Anomaly Score Gap."""
    # [Log Suppress]
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        if mode == 'OFF':
            # Frequency Stream Only (Semantic OFF)
            model = WEDGE_Net(use_semantic=False).to(DEVICE)
            base_dir = DIR_OFF
        else:
            # Full WEDGE-Net (Semantic ON)
            model = WEDGE_Net(use_semantic=True).to(DEVICE)
            base_dir = DIR_ON
    
    pt_path = get_model_path(base_dir, category)
    if pt_path is None: return None

    print(f"    [{mode}] Found at: {os.path.basename(pt_path)}")

    # Load Checkpoint
    try:
        checkpoint = torch.load(pt_path, map_location=DEVICE)
        bank = checkpoint['memory_bank'] if isinstance(checkpoint, dict) else checkpoint
        bank = bank.to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"Error loading {pt_path}: {e}")
        return None

    # Load Test Data
    try:
        ds = MVTecDataset(root_dir=config.DATA_PATH, category=category, phase='test')
        loader = DataLoader(ds, batch_size=1, shuffle=False)
    except: return None

    normal_scores = []
    defect_scores = []

    with torch.no_grad():
        for img, label, _, _ in loader:
            img = img.to(DEVICE)
            
            # [CRITICAL FIX] Always use model() forward pass
            # This ensures Frequency Stream is applied consistently with training.
            # If mode=='OFF', model is initialized with use_semantic=False,
            # so it naturally produces Frequency-Only features.
            features, _ = model(img)
            
            # Anomaly Scoring
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
        
    return np.mean(defect_scores) - np.mean(normal_scores)

# ==============================================================================
# 3. Main Execution Flow
# ==============================================================================
def main():
    print(f"\n" + "="*60)
    print(f" [Discussion] Score Gap Analysis (Table 7)")
    print(f" Target Ratio  : {TARGET_RATIO}")
    print(f" Semantic OFF  : {DIR_OFF}")
    print(f" Semantic ON   : {DIR_ON}")
    print("="*60)
    
    results = []
    tex_gaps_off, tex_gaps_on = [], []
    obj_gaps_off, obj_gaps_on = [], []

    for cat in ALL_CATEGORIES:
        group_type = "Texture" if cat in TEXTURE_CLASSES else "Object"
        print(f" >> [Processing] Category: {cat.upper()} ({group_type})")
        
        gap_off = calculate_gap(cat, 'OFF')
        gap_on = calculate_gap(cat, 'ON')
        
        if gap_off is not None and gap_on is not None:
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

    print("\n[Info] Analysis Complete.")

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
        writer.writerow({}) 
        writer.writerows(avg_rows)

    print(f"✅ CSV Saved to: {CSV_PATH}")
    print("-" * 60)
    for row in avg_rows:
        print(f" >> {row['Category']}: Gap {row['Gap_OFF']:.4f} -> {row['Gap_ON']:.4f} (Imp: {row['Improvement(%)']}%)")
    print("-" * 60)

if __name__ == "__main__":
    main()
