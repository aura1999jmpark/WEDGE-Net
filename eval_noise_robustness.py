"""
Noise Robustness Evaluation Script for WEDGE-Net vs PatchCore

Updates:
1. Applied Gaussian Smoothing (sigma=4.0) and Interpolation.
2. Added 'Denormalization' step before adding noise.
3. Added Error Handling for missing PatchCore models (Soft Fail).
4. Added Automatic Graph Plotting (Figure Generation).

Usage:
    Ensure 'config.py' is configured correctly before running.
    python eval_noise_robustness.py
"""

import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Added for plotting
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

# ImageNet Mean/Std for Normalization/Denormalization
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

def get_ratio_folder_name(ratio_str):
    """Converts sampling ratio to folder name."""
    try:
        ratio_float = float(ratio_str)
        if ratio_float == 1.0:
            return "100pct"
        percentage = int(ratio_float * 100)
        return f"{percentage}pct"
    except ValueError:
        return "10pct"

RATIO_FOLDER = get_ratio_folder_name(config.SAMPLING_RATIO)

# Path to WEDGE-Net memory bank
PATH_WEDGE = os.path.join(
    config.OurModel_DIR, 
    RATIO_FOLDER, 
    f"model_data_{CATEGORY}_{RATIO_FOLDER}.pt"
)

# Path to PatchCore memory bank (Flexible)
PATH_PC = getattr(config, 'CompareModel_DIR', "patch_core_pt") + f"/model_data_{CATEGORY}.pt"

# ==============================================================================
# Helper Functions
# ==============================================================================
class UnNormalize(object):
    """Restores normalized tensor to original [0, 1] range."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

# Global Transforms
unorm = UnNormalize(MEAN, STD)
norm = transforms.Normalize(MEAN, STD)

def add_noise(images, noise_level):
    """Adds Gaussian noise to [0, 1] images."""
    if noise_level == 0:
        return images
    noise = torch.randn_like(images) * (noise_level / 255.0)
    return torch.clamp(images + noise, 0, 1)

def get_pc_model():
    """Loads frozen WideResNet50."""
    model = models.wide_resnet50_2(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.to(DEVICE)
    model.eval()
    return model

def extract_pc_features_spatial(model, x):
    """Extracts features maintaining spatial dimensions (B, C, H, W)."""
    features = []
    def hook(module, input, output): features.append(output)
    
    h1 = model.layer2.register_forward_hook(hook)
    h2 = model.layer3.register_forward_hook(hook)
    
    _ = model(x)
    
    h1.remove()
    h2.remove()
    
    f1 = F.avg_pool2d(features[0], 3, 1, 1)
    f2 = F.avg_pool2d(features[1], 3, 1, 1)
    f2 = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=False)
    
    z = torch.cat([f1, f2], dim=1) 
    return z

def apply_smoothing(score_map_tensor, target_size):
    """Applies Interpolation -> Gaussian Smoothing -> Max Score"""
    score_map = F.interpolate(
        score_map_tensor, 
        size=(target_size, target_size), 
        mode='bilinear', 
        align_corners=False
    )
    score_map_np = score_map.squeeze().cpu().numpy()
    score_map_smooth = gaussian_filter(score_map_np, sigma=4.0)
    return score_map_smooth.max()

# ==============================================================================
# Main Evaluation Loop
# ==============================================================================
def run_full_evaluation():
    print(f"[Info] Starting Noise Robustness Eval: WEDGE-Net vs PatchCore | Category: {CATEGORY.upper()}")
    print(f"[Info] Target Folder: {RATIO_FOLDER} (Ratio: {config.SAMPLING_RATIO})")
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Load Models and Memory Banks
    print("Loading models...", end=" ")
    
    # WEDGE-Net
    if not os.path.exists(PATH_WEDGE):
        print(f"\n[Error] WEDGE-Net file not found at: {PATH_WEDGE}")
        return
    model_wedge = WEDGE_Net(use_semantic=True).to(DEVICE).eval()
    ckpt_w = torch.load(PATH_WEDGE, map_location=DEVICE)
    bank_wedge = ckpt_w['memory_bank'].to(DEVICE) if isinstance(ckpt_w, dict) else ckpt_w.to(DEVICE)

    # PatchCore (Conditional Load)
    has_pc = False
    model_pc = None
    bank_pc = None
    
    if os.path.exists(PATH_PC):
        try:
            model_pc = get_pc_model()
            ckpt_p = torch.load(PATH_PC, map_location=DEVICE)
            bank_pc = ckpt_p['memory_bank'].to(DEVICE) if isinstance(ckpt_p, dict) else ckpt_p.to(DEVICE)
            has_pc = True
            print("(PatchCore Loaded)", end=" ")
        except Exception as e:
            print(f"(PatchCore Load Failed: {e})", end=" ")
    else:
        print("(PatchCore Skipped)", end=" ")
        
    print("Done.")

    # 2. Prepare Dataset
    test_dataset = MVTecDataset(root_dir=config.DATA_PATH, category=CATEGORY, phase='test')
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4) 
    
    print(f"[Info] Total Test Images: {len(test_dataset)}")

    results = []
    noise_levels = [0, 10, 20, 30, 40]

    # 3. Run Inference
    for nl in noise_levels:
        print(f"Processing Noise Level {nl}...", end="\r")
        s_wedge, s_pc, labels = [], [], []
        
        for img, label, _, _ in loader:
            img = img.to(DEVICE)
            
            # --- [Noise Injection] ---
            # 1. Un-normalize
            img_raw = unorm(img.clone().squeeze(0)).unsqueeze(0) 
            
            # 2. Add Noise
            img_noisy_raw = add_noise(img_raw, nl)
            
            # 3. Re-normalize
            img_final = norm(img_noisy_raw.squeeze(0)).unsqueeze(0)
            # -------------------------
            
            # --- WEDGE-Net Inference ---
            with torch.no_grad():
                fw, _ = model_wedge(img_final) 
                B, C, H, W = fw.shape
                
                fw_flat = fw.view(C, -1).permute(1, 0)
                dist_w = torch.cdist(fw_flat, bank_wedge)
                min_dist_w, _ = torch.min(dist_w, dim=1)
                
                score_map_w = min_dist_w.view(H, W).unsqueeze(0).unsqueeze(0)
                score_w = apply_smoothing(score_map_w, IMAGE_SIZE)
                s_wedge.append(score_w)
            
            # --- PatchCore Inference (Only if available) ---
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
                s_pc.append(0.0) # Dummy score
                
            labels.append(label.item())
        
        # Calculate AUROC
        try:
            auc_w = roc_auc_score(labels, s_wedge) * 100
            auc_p = roc_auc_score(labels, s_pc) * 100 if has_pc else 0.0
        except ValueError:
            auc_w, auc_p = 0.0, 0.0
            
        gap = auc_w - auc_p
        
        # print(f"Noise {nl:<2} | WEDGE: {auc_w:.2f}% | Gap: {gap:+.2f}")
        results.append({'Noise': nl, f'WEDGE({RATIO_FOLDER})': auc_w, 'PC(100%)': auc_p, 'Gap': gap})

    print(f"\n[Result] Evaluation Completed.")

    # 4. Save CSV Results
    df = pd.DataFrame(results)
    csv_name = f"final_noise_comparison_smoothed_{CATEGORY}_{RATIO_FOLDER}.csv"
    save_path = os.path.join(SAVE_DIR, csv_name)
    df.to_csv(save_path, index=False)
    print(f"[Info] CSV Saved: {save_path}")

    # 5. [Added] Plot and Save Graph
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
    plt.ylim(50, 105) # AUROC range
    
    fig_name = f"figure5_noise_robustness_{CATEGORY}_{RATIO_FOLDER}.png"
    plt.savefig(os.path.join(SAVE_DIR, fig_name), dpi=300, bbox_inches='tight')
    print(f"[Info] Graph Saved: {os.path.join(SAVE_DIR, fig_name)}")

if __name__ == '__main__':
    run_full_evaluation()