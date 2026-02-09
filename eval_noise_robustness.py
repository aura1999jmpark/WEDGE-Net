"""
Noise Robustness Evaluation Script for WEDGE-Net vs PatchCore

Updates:
1. Applied Gaussian Smoothing (sigma=4.0) and Interpolation.
2. Added 'Denormalization' step before adding noise.
3. Added Error Handling for missing PatchCore models (Soft Fail).
4. Added Automatic Graph Plotting (Figure Generation).
5. Auto-detects PatchCore model files regardless of naming convention.

Usage:
    Ensure 'config.py' is configured correctly before running.
    python eval_noise_robustness.py
"""
import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
import random
# Import project modules
import config 
from model import WEDGE_Net
from dataset import MVTecDataset

# ==============================================================================
# Configuration & Setup
# ==============================================================================
CATEGORY = config.CATEGORY
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = config.SAVE_DIR
IMAGE_SIZE = getattr(config, 'IMAGE_SIZE', 224)

# ImageNet Mean/Std
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ------------------------------------------------------------------------------
# [Fix 1] Correct Ratio Parsing (Supports 0.1%)
# ------------------------------------------------------------------------------
def get_ratio_folder_name(ratio_input):
    """Safely converts ratio input to folder name (e.g., 0.001 -> 0_1pct)."""
    raw = str(ratio_input).lower()
    
    if raw == 'all':
        return "1pct" # Default to Main Method
        
    try:
        val = float(raw)
        if abs(val - 0.001) < 1e-6: return "0_1pct" # 0.1% Fix
        elif abs(val - 0.01) < 1e-6: return "1pct"  # 1%
        elif abs(val - 0.1) < 1e-6: return "10pct"  # 10%
        elif abs(val - 1.0) < 1e-6: return "100pct" # 100%
        else:
            # Fallback for other numbers
            return f"{int(val * 100)}pct"
    except ValueError:
        return "1pct" # Safe fallback

RATIO_FOLDER = get_ratio_folder_name(config.SAMPLING_RATIO)
print(f"⚙️ Config Ratio: {config.SAMPLING_RATIO} -> Target Folder: {RATIO_FOLDER}")

# ------------------------------------------------------------------------------
# [Fix 2] Smart File Finder (For PatchCore & WEDGE-Net)
# ------------------------------------------------------------------------------
def find_best_model_path(base_dir, category, target_str=None):
    """Scans directory to find the best matching .pt file."""
    if not os.path.exists(base_dir): return None
    
    all_files = [f for f in os.listdir(base_dir) if f.endswith('.pt')]
    # Filter by category name
    candidates = [f for f in all_files if category in f]
    
    if not candidates: return None
    
    # Priority check (e.g., find '10pct' specifically)
    if target_str:
        priority = [f for f in candidates if target_str in f]
        if priority: return os.path.join(base_dir, priority[0])
    
    # Default: Shortest filename usually implies the cleanest/standard one
    candidates.sort(key=len)
    return os.path.join(base_dir, candidates[0])

# --- Path Definition ---

# 1. WEDGE-Net Path
wedge_base = os.path.join(config.OurModel_DIR, RATIO_FOLDER)
PATH_WEDGE = find_best_model_path(wedge_base, CATEGORY)
if PATH_WEDGE is None:
    # Fallback to hardcoded style just in case
    PATH_WEDGE = os.path.join(wedge_base, f"model_data_{CATEGORY}_{RATIO_FOLDER}.pt")

# 2. PatchCore Path (Smart Search)
pc_base = getattr(config, 'CompareModel_DIR', "patch_core_pt")
# Try to find '10pct' version first (fair comparison), else any version
PATH_PC = find_best_model_path(pc_base, CATEGORY, target_str="10pct")
if PATH_PC is None:
    PATH_PC = find_best_model_path(pc_base, CATEGORY) # Find anything

# ==============================================================================
# Helper Functions
# ==============================================================================
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std
    def __call__(self, tensor):
        # Broadcasting Fix
        if tensor.ndim == 3: tensor = tensor.unsqueeze(0)
        t_copy = tensor.clone()
        m = torch.tensor(self.mean).view(1, 3, 1, 1).to(tensor.device)
        s = torch.tensor(self.std).view(1, 3, 1, 1).to(tensor.device)
        t_copy.mul_(s).add_(m)
        return t_copy

unorm = UnNormalize(MEAN, STD)
norm = transforms.Normalize(MEAN, STD)

def add_noise(images, noise_level):
    if noise_level == 0: return images
    noise = torch.randn_like(images) * (noise_level / 255.0)
    return torch.clamp(images + noise, 0, 1)

def get_pc_model():
    model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    for param in model.parameters(): param.requires_grad = False
    model.to(DEVICE).eval()
    return model

def extract_pc_features_spatial(model, x):
    features = []
    def hook(module, input, output): features.append(output)
    h1 = model.layer2.register_forward_hook(hook)
    h2 = model.layer3.register_forward_hook(hook)
    _ = model(x)
    h1.remove(); h2.remove()
    
    f1 = F.avg_pool2d(features[0], 3, 1, 1)
    f2 = F.avg_pool2d(features[1], 3, 1, 1)
    f2 = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=False)
    z = torch.cat([f1, f2], dim=1) 
    return z

def apply_smoothing(score_map_tensor, target_size):
    score_map = F.interpolate(score_map_tensor, size=(target_size, target_size), mode='bilinear', align_corners=False)
    score_map_np = score_map.squeeze().cpu().numpy()
    score_map_smooth = gaussian_filter(score_map_np, sigma=4.0)
    return score_map_smooth.max()

# ==============================================================================
# Main Evaluation Loop
# ==============================================================================
def run_full_evaluation():
    print(f"[Info] Starting Noise Robustness Eval: WEDGE-Net vs PatchCore | Category: {CATEGORY.upper()}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Load Models
    print("Loading models...", end=" ")
    
    # WEDGE-Net
    if not PATH_WEDGE or not os.path.exists(PATH_WEDGE):
        print(f"\n[Error] WEDGE-Net file not found via smart search.")
        return
    
    try:
        print(f"\n   -> WEDGE-Net: {os.path.basename(PATH_WEDGE)}")
        model_wedge = WEDGE_Net(use_semantic=True).to(DEVICE).eval()
        ckpt_w = torch.load(PATH_WEDGE, map_location=DEVICE)
        bank_wedge = ckpt_w['memory_bank'].to(DEVICE) if isinstance(ckpt_w, dict) else ckpt_w.to(DEVICE)
    except Exception as e:
        print(f"[Error] Failed to load WEDGE-Net: {e}")
        return

    # PatchCore
    has_pc = False
    model_pc, bank_pc = None, None
    
    if PATH_PC and os.path.exists(PATH_PC):
        try:
            print(f"   -> PatchCore: {os.path.basename(PATH_PC)}")
            model_pc = get_pc_model()
            ckpt_p = torch.load(PATH_PC, map_location=DEVICE)
            bank_pc = ckpt_p['memory_bank'].to(DEVICE) if isinstance(ckpt_p, dict) else ckpt_p.to(DEVICE)
            has_pc = True
        except Exception as e:
            print(f"   -> PatchCore Load Failed: {e}")
    else:
        print("   -> PatchCore: Not Found (Skipping comparison)")
        
    print("Done.")

    # 2. Prepare Dataset
    try:
        test_dataset = MVTecDataset(root_dir=config.DATA_PATH, category=CATEGORY, phase='test')
        loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4) 
    except:
        print(f"[Error] Dataset load failed for {CATEGORY}")
        return
        
    print(f"[Info] Total Test Images: {len(test_dataset)}")

    results = []
    noise_levels = [0, 10, 20, 30, 40]

    # 3. Run Inference
    for nl in noise_levels:
        print(f"Processing Noise Level {nl}...", end="\r")
        s_wedge, s_pc, labels = [], [], []
        
        for img, label, _, _ in loader:
            img = img.to(DEVICE)
            
            # --- Noise Injection ---
            img_raw = unorm(img.clone().squeeze(0)).unsqueeze(0) 
            img_noisy_raw = add_noise(img_raw, nl)
            img_final = norm(img_noisy_raw.squeeze(0)).unsqueeze(0)
            
            # --- WEDGE-Net ---
            with torch.no_grad():
                fw, _ = model_wedge(img_final) 
                B, C, H, W = fw.shape
                fw_flat = fw.view(C, -1).permute(1, 0)
                dist_w = torch.cdist(fw_flat, bank_wedge)
                min_dist_w, _ = torch.min(dist_w, dim=1)
                score_map_w = min_dist_w.view(H, W).unsqueeze(0).unsqueeze(0)
                score_w = apply_smoothing(score_map_w, IMAGE_SIZE)
                s_wedge.append(score_w)
            
            # --- PatchCore ---
            if has_pc:
                with torch.no_grad():
                    fp = extract_pc_features_spatial(model_pc, img_final)
                    B, C, H, W = fp.shape
                    fp_flat = fp.view(C, -1).permute(1, 0)
                    dist_p = torch.cdist(fp_flat, bank_pc)
                    min_dist_p, _ = torch.min(dist_p, dim=1)
                    score_map_p = min_dist_p.view(H, W).unsqueeze(0).unsqueeze(0)
                    score_p = apply_smoothing(score_map_p, IMAGE_SIZE)
                    s_pc.append(score_p)
            else:
                s_pc.append(0.0) 
                
            labels.append(label.item())
        
        # Calculate AUROC
        try:
            auc_w = roc_auc_score(labels, s_wedge) * 100
            auc_p = roc_auc_score(labels, s_pc) * 100 if has_pc else 0.0
        except ValueError:
            auc_w, auc_p = 0.0, 0.0
            
        gap = auc_w - auc_p
        results.append({'Noise': nl, f'WEDGE({RATIO_FOLDER})': auc_w, 'PC(100%)': auc_p, 'Gap': gap})

    print(f"\n[Result] Evaluation Completed.")

    # 4. Save CSV
    df = pd.DataFrame(results)
    csv_name = f"full_noise_comparison_smoothed_{CATEGORY}_{RATIO_FOLDER}.csv"
    save_path = os.path.join(SAVE_DIR, csv_name)
    df.to_csv(save_path, index=False)
    print(f"[Info] CSV Saved: {save_path}")

    # 5. Plot Graph
    plt.figure(figsize=(10, 8))
    plt.plot(df['Noise'], df[f'WEDGE({RATIO_FOLDER})'], marker='o', markersize=12, 
             linewidth=3, label=f'WEDGE-Net ({RATIO_FOLDER})', color='#1f77b4')
    
    if has_pc:
        plt.plot(df['Noise'], df['PC(100%)'], marker='s', markersize=12, 
                 linewidth=3, linestyle='--', label='PatchCore (100% Full)', color='#d62728')

    plt.title(f"Robustness against Gaussian Noise ({CATEGORY.upper()})", fontsize=16, fontweight='bold')
    plt.xlabel("Noise Level (sigma)", fontsize=14)
    plt.ylabel("AUROC Score (%)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.ylim(50, 105) 
    
    fig_name = f"figure5_noise_robustness_{CATEGORY}_{RATIO_FOLDER}.png"
    plt.savefig(os.path.join(SAVE_DIR, fig_name), dpi=300, bbox_inches='tight')
    print(f"[Info] Graph Saved: {os.path.join(SAVE_DIR, fig_name)}")

if __name__ == '__main__':
    run_full_evaluation()
