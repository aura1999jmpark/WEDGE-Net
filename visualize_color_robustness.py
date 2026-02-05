import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from scipy.ndimage import gaussian_filter
from model import WEDGE_Net
from dataset import MVTecDataset 
import config

# ==============================================================================
# 1. Settings & Constants
# ==============================================================================
BASE_DIR = config.SAVE_DIR
CATEGORY = config.CATEGORY
# Intensity of experiment: 0 (Original) ~ 2.0 (Severe)
INTENSITIES = [0.5, 1.0, 1.5, 2.0] 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ImageNet Mean/Std
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# [Fix] Determine Path Suffix based on Config Ratio
target_ratio_str = str(config.SAMPLING_RATIO)

if '0.01' in target_ratio_str:
    ratio_suffix = "1pct"
elif '0.1' in target_ratio_str:
    ratio_suffix = "10pct"
elif '1.0' in target_ratio_str or '1' == target_ratio_str:
    ratio_suffix = "100pct"
else:
    # Fallback default
    ratio_suffix = "10pct"
    print(f"⚠️ Warning: Config ratio '{config.SAMPLING_RATIO}' not recognized. Defaulting to '10pct'.")

print(f"🎯 Config Ratio: {config.SAMPLING_RATIO} -> Target Folder: {ratio_suffix}")
# WEDGE-Net Path (Dynamic)
PATH_WEDGE = os.path.join(
    config.OurModel_DIR, 
    ratio_suffix, 
    f"model_data_{CATEGORY}_{ratio_suffix}.pt"
)


# PatchCore Path (Handle flexibly)
PATH_PC = getattr(config, 'CompareModel_DIR', "patch_core_pt") + f"/model_data_{CATEGORY}.pt"

# ==============================================================================
# 2. Utility Classes & Functions
# ==============================================================================
class UnNormalize(object):
    """Restores normalized tensor to original image (0~1)"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        tensor_copy = tensor.clone()
        for t, m, s in zip(tensor_copy, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor_copy

# Global transformation objects
unorm = UnNormalize(IMAGENET_MEAN, IMAGENET_STD)
norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

def tensor_to_img_numpy(tensor):
    """Converts (C, H, W) 0~1 tensor to (H, W, C) 0~255 numpy array"""
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return img

def apply_color_jitter(image_tensor, factor):
    """
    Applies color distortion (Brightness, Contrast, Saturation, Hue).
    Input: (B, C, H, W) range [0, 1] (Un-normalized)
    """
    jittered = image_tensor.clone()
    
    if factor == 0:
        return jittered

    # Apply transformations sequentially based on intensity factor
    jittered = transforms.functional.adjust_brightness(jittered, brightness_factor=1.0 + (factor * 0.3)) 
    jittered = transforms.functional.adjust_contrast(jittered, contrast_factor=1.0 + (factor * 0.5))
    jittered = transforms.functional.adjust_saturation(jittered, saturation_factor=1.0 + (factor * 0.5))
    
    hue_factor = min(0.1 * factor, 0.5)
    jittered = transforms.functional.adjust_hue(jittered, hue_factor=hue_factor)
    
    return torch.clamp(jittered, 0, 1)

def get_maps(model_pc, bank_pc, model_wd, bank_wd, img_tensor):
    """
    Extracts Anomaly Maps for PatchCore and WEDGE-Net.
    Skips calculation if PatchCore model is None.
    """
    # Original image dimensions
    H, W = img_tensor.shape[2], img_tensor.shape[3]
    map_pc = None 

    # --- PatchCore Inference (Run only if model exists) ---
    if model_pc is not None and bank_pc is not None:
        try:
            features = []
            def hook(module, input, output): features.append(output)
            h1 = model_pc.layer2.register_forward_hook(hook)
            h2 = model_pc.layer3.register_forward_hook(hook)
            
            _ = model_pc(img_tensor)
            
            h1.remove(); h2.remove()
            
            f1 = F.avg_pool2d(features[0], 3, 1, 1)
            f2 = F.avg_pool2d(features[1], 3, 1, 1)
            f2 = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=False)
            
            z = torch.cat([f1, f2], dim=1) 
            
            # Dynamic feature map size extraction
            _, _, h_pc, w_pc = z.shape
            embedding = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
            
            dists = torch.cdist(embedding, bank_pc)
            min_dist, _ = dists.min(dim=1)
            
            # 1. Reshape to feature map size
            map_pc = min_dist.reshape(h_pc, w_pc).unsqueeze(0).unsqueeze(0)
            # 2. Upsample to original size
            map_pc = F.interpolate(map_pc, size=(H, W), mode='bilinear', align_corners=False)
            map_pc = map_pc.squeeze().cpu().numpy()
        except Exception as e:
            print(f"⚠️ Error during PatchCore inference: {e}")
            map_pc = None
    else:
        # Return zeros if model is missing (for visualization placeholder)
        map_pc = np.zeros((H, W))

    # --- WEDGE-Net Inference (Fixed Logic) ---
    features_wd, _ = model_wd(img_tensor)
    
    # [Important] Get actual feature map size (usually 28x28)
    B, C, h_wd, w_wd = features_wd.shape 
    
    flat_wd = features_wd.view(C, -1).permute(1, 0)
    dists_wd = torch.cdist(flat_wd, bank_wd)
    min_dist_wd, _ = dists_wd.min(dim=1)
    
    # [Fix] Reshape to (h_wd, w_wd) first, then interpolate
    map_wd = min_dist_wd.reshape(h_wd, w_wd).unsqueeze(0).unsqueeze(0)
    map_wd = F.interpolate(map_wd, size=(H, W), mode='bilinear', align_corners=False)
    map_wd = map_wd.squeeze().cpu().numpy()
    
    return map_pc, map_wd

# ==============================================================================
# 3. Main Execution Function
# ==============================================================================
def run_color_experiment():
    print(f"🎨 Generating Color Robustness Figure for {CATEGORY.upper()}...")
    os.makedirs(BASE_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Load Models
    # ---------------------------------------------------------
    # [Modified] Check PatchCore Existence
    HAS_PC = False
    pc_model = None
    pc_bank = None

    print(f"🔍 Checking PatchCore: {PATH_PC}")
    if os.path.exists(PATH_PC):
        try:
            print("   -> Found! Loading PatchCore...")
            pc_model = models.wide_resnet50_2(pretrained=True).to(DEVICE).eval()
            for p in pc_model.parameters(): p.requires_grad = False
            pc_ckpt = torch.load(PATH_PC, map_location=DEVICE)
            pc_bank = pc_ckpt['memory_bank'].to(DEVICE) if isinstance(pc_ckpt, dict) else pc_ckpt.to(DEVICE)
            HAS_PC = True
        except Exception as e:
            print(f"   -> Error loading PatchCore: {e}. Skipping...")
            HAS_PC = False
    else:
        print("   -> Not found. Skipping comparison (Running WEDGE-Net only).")
    
    # Load WEDGE-Net (Strict Check)
    print(f"Loading WEDGE-Net (10%): {PATH_WEDGE}")
    if not os.path.exists(PATH_WEDGE): 
        return print(f"🚨 Critical Error: WEDGE-Net model not found at {PATH_WEDGE}")
        
    wedge_model = WEDGE_Net(use_semantic=True).to(DEVICE).eval()
    wedge_ckpt = torch.load(PATH_WEDGE, map_location=DEVICE)
    wedge_bank = wedge_ckpt['memory_bank'].to(DEVICE) if isinstance(wedge_ckpt, dict) else wedge_ckpt.to(DEVICE)
    
    # Downsampling for visualization speed
    if wedge_bank.shape[0] > 10000:
        print("⚡ Downsampling large memory bank...")
        torch.manual_seed(42)
        idx = torch.randperm(wedge_bank.shape[0])[:int(wedge_bank.shape[0]*0.1)]
        wedge_bank = wedge_bank[idx]

    # ---------------------------------------------------------
    # 2. Select Normal Image
    # ---------------------------------------------------------
    dataset = MVTecDataset(root_dir=config.DATA_PATH, category=CATEGORY, phase='test')
    target_img = None
    target_name = ""
    
    for i in range(len(dataset)):
        img, label, _, path = dataset[i]
        lbl_val = label.item() if hasattr(label, 'item') else label
        
        if lbl_val == 0: # Normal
            target_img = img.to(DEVICE)
            path_str = path[0] if isinstance(path, (list, tuple)) else path
            target_name = os.path.basename(path_str)
            print(f"✅ Selected Normal Image: {target_name}")
            break
            
    if target_img is None: return print("🚨 No normal image found in dataset")

    # ---------------------------------------------------------
    # 3. Generate Figure
    # ---------------------------------------------------------
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # 4. Loop (Color Intensity)
    for i, intensity in enumerate(INTENSITIES):
        # [Pipeline] UnNorm -> Color Jitter -> ReNorm
        img_raw = unorm(target_img).to(DEVICE) 
        
        # Apply Color Jitter
        distorted_raw = apply_color_jitter(img_raw, intensity)
        
        # Re-normalize for model input
        distorted_model = norm(distorted_raw.squeeze(0)).unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            raw_pc, raw_wd = get_maps(pc_model, pc_bank, wedge_model, wedge_bank, distorted_model)
            
        # [Post-process] Gaussian Smoothing (Unified)
        if HAS_PC: raw_pc = gaussian_filter(raw_pc, sigma=4)
        raw_wd = gaussian_filter(raw_wd, sigma=4)
        
        # Convert to Numpy for display
        clean_disp = tensor_to_img_numpy(img_raw.squeeze(0))
        distorted_disp = tensor_to_img_numpy(distorted_raw.squeeze(0))

        v_min = raw_wd.min()
        v_max = raw_wd.max()
        if HAS_PC:
            v_min = min(raw_pc.min(), v_min)
            v_max = max(raw_pc.max(), v_max)

        # (a) Original
        ax = axes[i, 0]
        ax.imshow(clean_disp)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"Shift Level {intensity}", fontsize=18, fontweight='bold', labelpad=10)

        # (b) Color Shifted Input
        ax = axes[i, 1]
        ax.imshow(distorted_disp)
        ax.set_xticks([]); ax.set_yticks([])

        # (c) PatchCore (Conditional)
        ax = axes[i, 2]
        if HAS_PC:
            ax.imshow(distorted_disp)
            ax.imshow(raw_pc, cmap='jet', alpha=0.5, vmin=v_min, vmax=v_max)
            ax.text(0.95, 0.05, f"{raw_pc.max():.3f}", color='white', fontsize=20, fontweight='bold',
                       ha='right', va='bottom', transform=ax.transAxes,
                       bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=3))
        else:
            # Placeholder for missing model
            ax.set_facecolor('lightgray')
            ax.text(0.5, 0.5, "Checkpoint\nNot Found\n(Skipped)", 
                       ha='center', va='center', fontsize=14, color='gray')
        ax.set_xticks([]); ax.set_yticks([])

        # (d) WEDGE-Net
        ax = axes[i, 3]
        ax.imshow(distorted_disp)
        ax.imshow(raw_wd, cmap='jet', alpha=0.5, vmin=v_min, vmax=v_max)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.95, 0.05, f"{raw_wd.max():.3f}", color='white', fontsize=20, fontweight='bold',
                   ha='right', va='bottom', transform=ax.transAxes,
                   bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=3))

    # Header Titles
    cols = ['(a) Original', '(b) Color Shifted', '(c) PatchCore (Ref)', '(d) WEDGE-Net (Ours)']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=22, fontweight='bold', pad=15)

    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, f"Figure_Color_Robustness_{CATEGORY}_Final.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Color experiment completed! Saved to: {save_path}")
    # plt.show() # Uncomment if you want to see the plot window

if __name__ == '__main__':
    run_color_experiment()
