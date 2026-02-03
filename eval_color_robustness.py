"""
Color Robustness Evaluation Script for WEDGE-Net vs PatchCore

Metrics: AUROC under varying Color Jitter intensities.
Process: UnNormalize -> Color Jitter -> ReNormalize -> Inference -> Smoothing -> AUROC
"""

import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Added for plotting
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
# [Update] Dynamic Path Configuration based on config.SAMPLING_RATIO
# ------------------------------------------------------------------------------
def get_ratio_label(ratio):
    """Converts numeric/string ratio to folder name (e.g., '0.1' -> '10pct')"""
    r = str(ratio)
    if r == '0.1': return '10pct'
    if r == '0.01': return '1pct'
    if r == '1.0': return '100pct'
    return '10pct' # Default fallback

RATIO_LABEL = get_ratio_label(config.SAMPLING_RATIO)

PATH_WEDGE = os.path.join(
    config.OurModel_DIR, 
    RATIO_LABEL, 
    f"model_data_{CATEGORY}_{RATIO_LABEL}.pt"
)

# PatchCore path (flexible handling)
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

def apply_color_jitter(image_tensor, factor):
    """
    Applies color jitter (Brightness, Contrast, Saturation, Hue).
    Same logic as the visualization script.
    """
    if factor == 0:
        return image_tensor
        
    jittered = image_tensor.clone()
    
    # Apply Brightness, Contrast, Saturation sequentially
    jittered = transforms.functional.adjust_brightness(jittered, brightness_factor=1.0 + (factor * 0.3)) 
    jittered = transforms.functional.adjust_contrast(jittered, contrast_factor=1.0 + (factor * 0.5))
    jittered = transforms.functional.adjust_saturation(jittered, saturation_factor=1.0 + (factor * 0.5))
    
    # Apply Hue
    hue_factor = min(0.1 * factor, 0.5)
    jittered = transforms.functional.adjust_hue(jittered, hue_factor=hue_factor)
    
    return torch.clamp(jittered, 0, 1)

def get_pc_model():
    """Loads Pretrained WideResNet50 for PatchCore"""
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
    
    z = torch.cat([f1, f2], dim=1) # (B, 1536, H, W)
    return z

def apply_smoothing(score_map_tensor, target_size):
    """Interpolation -> Gaussian Smoothing -> Max Score"""
    # 1. Interpolate
    score_map = F.interpolate(
        score_map_tensor, 
        size=(target_size, target_size), 
        mode='bilinear', 
        align_corners=False
    )
    
    # 2. Gaussian Smoothing
    score_map_np = score_map.squeeze().cpu().numpy()
    score_map_smooth = gaussian_filter(score_map_np, sigma=4.0)
    
    return score_map_smooth.max()

# ==============================================================================
# Main Evaluation Loop
# ==============================================================================
def run_color_evaluation():
    print(f"[Info] Starting Color Robustness Eval: WEDGE-Net vs PatchCore | Category: {CATEGORY.upper()}")
    print(f"[Info] Sampling Ratio: {config.SAMPLING_RATIO} -> Folder: {RATIO_LABEL}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Load Models and Memory Banks
    print("Loading models...", end=" ")
    
    # WEDGE-Net Load
    if not os.path.exists(PATH_WEDGE):
        print(f"\n[Error] WEDGE-Net file not found at: {PATH_WEDGE}")
        return
    model_wedge = WEDGE_Net(use_semantic=True).to(DEVICE).eval()
    ckpt_w = torch.load(PATH_WEDGE, map_location=DEVICE)
    bank_wedge = ckpt_w['memory_bank'].to(DEVICE) if isinstance(ckpt_w, dict) else ckpt_w.to(DEVICE)

    # PatchCore Load (Conditional)
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
    # Experiment Intensity: 0.0 (Clean) ~ 3.0 (Severe)
    intensities = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # 3. Run Inference Loop
    for intensity in intensities:
        print(f"Processing Intensity {intensity}...", end="\r")
        s_wedge, s_pc, labels = [], [], []
        
        for img, label, _, _ in loader:
            img = img.to(DEVICE)
            
            # --- [Correct Color Injection Pipeline] ---
            # 1. Un-normalize
            img_raw = unorm(img.clone().squeeze(0)).unsqueeze(0) 
            
            # 2. Apply Color Jitter (0~1 range)
            img_distorted_raw = apply_color_jitter(img_raw, intensity)
            
            # 3. Re-normalize
            img_final = norm(img_distorted_raw.squeeze(0)).unsqueeze(0)
            # ------------------------------------------
            
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
        
        results.append({'Intensity': intensity, f'WEDGE({RATIO_LABEL})': auc_w, 'PC(100%)': auc_p, 'Gap': gap})

    print(f"\n[Result] Evaluation Completed.")

    # 4. Save CSV Results
    df = pd.DataFrame(results)
    csv_name = f"final_color_comparison_{CATEGORY}_{RATIO_LABEL}.csv"
    save_path = os.path.join(SAVE_DIR, csv_name)
    df.to_csv(save_path, index=False)
    print(f"[Info] CSV Saved: {save_path}")

    # 5. [Added] Save Graph Image (Figure 6 Style)
    plt.figure(figsize=(10, 8))
    plt.plot(df['Intensity'], df[f'WEDGE({RATIO_LABEL})'], marker='o', markersize=12, 
             linewidth=3, label=f'WEDGE-Net ({RATIO_LABEL})', color='#1f77b4')
    
    if has_pc:
        plt.plot(df['Intensity'], df['PC(100%)'], marker='s', markersize=12, 
                 linewidth=3, linestyle='--', label='PatchCore (100% Full)', color='#d62728')

    plt.title(f"Robustness against Color Jitter ({CATEGORY.upper()})", fontsize=16, fontweight='bold')
    plt.xlabel("Color Jitter Intensity (factor)", fontsize=14)
    plt.ylabel("AUROC Score (%)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    fig_name = f"figure6_color_robustness_{CATEGORY}_{RATIO_LABEL}.png"
    plt.savefig(os.path.join(SAVE_DIR, fig_name), dpi=300, bbox_inches='tight')
    print(f"[Info] Graph Saved: {os.path.join(SAVE_DIR, fig_name)}")

if __name__ == '__main__':
    run_color_evaluation()