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
import glob
import sys  

# ==============================================================================
# 1. Configuration & Global Settings
# ==============================================================================
ALL_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

BASE_DIR = getattr(config, 'SAVE_DIR', 'WEDGE-Net')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NOISE_LEVELS = [0, 10, 20, 30] 
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# [Config] Ratio Parsing (Enhanced Message)
raw_ratio = str(getattr(config, 'SAMPLING_RATIO', '0.01')).lower()

if raw_ratio == 'all':
    print("\n" + "-"*60)
    print(" [Info] Configured SAMPLING_RATIO = 'all'.")
    print(" To ensure concise visualization (Figure 5), we focus on the Main Proposed Method (1%).")
    print(" Processing all ratios would generate redundant figures.")
    print(" -> Defaulting to '1pct' (1%) for this run.")
    print("-"*60 + "\n")
    ratio_suffix = "1pct"
else:
    try:
        val = float(raw_ratio)
        if abs(val - 0.001) < 1e-6: ratio_suffix = "0_1pct"
        elif abs(val - 0.01) < 1e-6: ratio_suffix = "1pct"
        elif abs(val - 0.1) < 1e-6: ratio_suffix = "10pct"
        elif abs(val - 1.0) < 1e-6: ratio_suffix = "100pct"
        else: ratio_suffix = "1pct"
    except ValueError:
        ratio_suffix = "1pct"

print(f"🎯 Target Ratio: {raw_ratio} -> Folder: '{ratio_suffix}'")

# ==============================================================================
# 2. Smart File Finder
# ==============================================================================
def find_best_model_path(base_dir, category, target_str=None):
    if not os.path.exists(base_dir): return None
    all_files = [f for f in os.listdir(base_dir) if f.endswith('.pt')]
    candidates = [f for f in all_files if category in f]
    if not candidates: return None
    if target_str:
        priority = [f for f in candidates if target_str in f]
        if priority: return os.path.join(base_dir, priority[0])
    candidates.sort(key=len)
    return os.path.join(base_dir, candidates[0])

# ==============================================================================
# 3. Utility Classes
# ==============================================================================
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std
    def __call__(self, tensor):
        if tensor.ndim == 3: tensor = tensor.unsqueeze(0)
        t_copy = tensor.clone()
        m = torch.tensor(self.mean).view(1, 3, 1, 1).to(tensor.device)
        s = torch.tensor(self.std).view(1, 3, 1, 1).to(tensor.device)
        t_copy.mul_(s).add_(m)
        return t_copy

unorm = UnNormalize(IMAGENET_MEAN, IMAGENET_STD)
norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

def tensor_to_img_numpy(tensor):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)

def get_maps(model_pc, bank_pc, model_wd, bank_wd, img_tensor):
    H, W = img_tensor.shape[2], img_tensor.shape[3]
    map_pc = None 

    # --- PatchCore Inference ---
    if model_pc and bank_pc is not None:
        try:
            features = []
            def hook(m, i, o): features.append(o)
            h1 = model_pc.layer2.register_forward_hook(hook)
            h2 = model_pc.layer3.register_forward_hook(hook)
            _ = model_pc(img_tensor)
            h1.remove(); h2.remove()
            
            f1 = F.avg_pool2d(features[0], 3, 1, 1)
            f2 = F.avg_pool2d(features[1], 3, 1, 1)
            f2 = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=False)
            z = torch.cat([f1, f2], dim=1) 
            
            _, _, h_pc, w_pc = z.shape
            embedding = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
            dists = torch.cdist(embedding, bank_pc)
            min_dist, _ = dists.min(dim=1)
            
            map_pc = min_dist.reshape(h_pc, w_pc).unsqueeze(0).unsqueeze(0)
            map_pc = F.interpolate(map_pc, size=(H, W), mode='bilinear', align_corners=False)
            map_pc = map_pc.squeeze().cpu().numpy()
        except: map_pc = np.zeros((H, W))
    else:
        map_pc = np.zeros((H, W))

    # --- WEDGE-Net Inference ---
    features_wd, _ = model_wd(img_tensor)
    B, C, h_wd, w_wd = features_wd.shape 
    flat_wd = features_wd.view(C, -1).permute(1, 0)
    dists_wd = torch.cdist(flat_wd, bank_wd)
    min_dist_wd, _ = dists_wd.min(dim=1)
    
    map_wd = min_dist_wd.reshape(h_wd, w_wd).unsqueeze(0).unsqueeze(0)
    map_wd = F.interpolate(map_wd, size=(H, W), mode='bilinear', align_corners=False)
    map_wd = map_wd.squeeze().cpu().numpy()
    
    return map_pc, map_wd

# ==============================================================================
# 4. Main Execution
# ==============================================================================
def run_noise_experiment():
    print("\n" + "="*60)
    print(" 🎨 Starting Noise Robustness Experiment (Figure 5)")
    print("="*60)
    os.makedirs(BASE_DIR, exist_ok=True)

    # [Logic] Use 'all' or Single Category
    raw_cat = getattr(config, 'CATEGORY', 'tile') 
    
    if str(raw_cat).lower() == 'all':
        print("\n" + "!"*60)
        print(" ⛔ [STOP] CATEGORY = 'all' detected!")
        print(" This script requires a SINGLE category to prevent file flooding.")
        print(" 👉 Action: Please change 'config.py' to: CATEGORY = 'tile' (or 'bottle', etc.)")
        print("!"*60 + "\n")
        sys.exit(0) # Terminate the script
    else:
        target_categories = [str(raw_cat)]
        print(f" [Mode] Processing SINGLE category: {raw_cat}")

    for category in target_categories:
        print(f"\n>> Processing Category: {category.upper()}")
        
        # 1. Path Setup
        wedge_base = os.path.join(getattr(config, 'OurModel_DIR', 'WEDGE-Net'), ratio_suffix)
        path_wedge = find_best_model_path(wedge_base, category)

        pc_base = getattr(config, 'CompareModel_DIR', "patch_core_pt")
        # Try to find '10pct' file first
        path_pc = find_best_model_path(pc_base, category, target_str="10pct") 
        if path_pc is None: path_pc = find_best_model_path(pc_base, category)

        if not path_wedge:
            print(f"   [Skip] WEDGE-Net model not found in {wedge_base}")
            continue

        # 2. Load Models
        HAS_PC = False
        pc_model, pc_bank = None, None
        if path_pc:
            try:
                print(f"   [Load] PatchCore: {os.path.basename(path_pc)}")
                pc_model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(DEVICE).eval()
                pc_ckpt = torch.load(path_pc, map_location=DEVICE)
                pc_bank = pc_ckpt['memory_bank'].to(DEVICE) if isinstance(pc_ckpt, dict) else pc_ckpt.to(DEVICE)
                HAS_PC = True
            except: 
                print(f"   [Warning] Failed to load PatchCore file.")
        
        try:
            print(f"   [Load] WEDGE-Net: {os.path.basename(path_wedge)}")
            wedge_model = WEDGE_Net(use_semantic=True).to(DEVICE).eval()
            wedge_ckpt = torch.load(path_wedge, map_location=DEVICE)
            wedge_bank = wedge_ckpt['memory_bank'].to(DEVICE) if isinstance(wedge_ckpt, dict) else wedge_ckpt.to(DEVICE)
        except Exception as e:
            print(f"   [Error] Failed to load WEDGE-Net: {e}")
            continue
        
        if wedge_bank.shape[0] > 10000:
            idx = torch.randperm(wedge_bank.shape[0])[:int(wedge_bank.shape[0]*0.5)]
            wedge_bank = wedge_bank[idx]

        # 3. Data Selection
        try:
            dataset = MVTecDataset(root_dir=config.DATA_PATH, category=category, phase='test')
        except:
            print(f"   [Error] Dataset not found for {category}")
            continue

        target_img = None
        for i in range(len(dataset)):
            img, label, _, path = dataset[i]
            if (label.item() if hasattr(label, 'item') else label) == 0:
                target_img = img.to(DEVICE)
                break
        
        if target_img is None:
            print("   [Skip] No normal image found.")
            continue

        # 4. Generate Figure
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        
        for i, noise_level in enumerate(NOISE_LEVELS):
            img_raw = unorm(target_img).to(DEVICE) 
            noise = torch.randn_like(img_raw) * (noise_level / 255.0)
            noisy_raw = torch.clamp(img_raw + noise, 0, 1)
            noisy_model = norm(noisy_raw.squeeze(0)).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                raw_pc, raw_wd = get_maps(pc_model, pc_bank, wedge_model, wedge_bank, noisy_model)
            
            if HAS_PC: raw_pc = gaussian_filter(raw_pc, sigma=4)
            raw_wd = gaussian_filter(raw_wd, sigma=4)
            
            clean_disp = tensor_to_img_numpy(img_raw.squeeze(0))
            noisy_disp = tensor_to_img_numpy(noisy_raw.squeeze(0))

            v_min = raw_wd.min()
            current_max = raw_wd.max()
            if HAS_PC:
                v_min = min(raw_pc.min(), v_min)
                current_max = max(raw_pc.max(), current_max)
            v_max = current_max if (current_max - v_min) > 0.5 else v_min + 0.5

            ax = axes[i, 0]
            ax.imshow(clean_disp)
            ax.set_ylabel(f"Noise $\sigma={noise_level}$", fontsize=16, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])

            ax = axes[i, 1]
            ax.imshow(noisy_disp)
            ax.set_xticks([]); ax.set_yticks([])

            ax = axes[i, 2]
            if HAS_PC:
                ax.imshow(noisy_disp)
                ax.imshow(raw_pc, cmap='jet', alpha=0.5, vmin=v_min, vmax=v_max)
                ax.text(0.95, 0.05, f"{raw_pc.max():.2f}", color='white', fontweight='bold', transform=ax.transAxes, ha='right', bbox=dict(facecolor='black', alpha=0.5))
            else:
                ax.set_facecolor('gray')
                ax.text(0.5, 0.5, "N/A", ha='center')
            ax.set_xticks([]); ax.set_yticks([])

            ax = axes[i, 3]
            ax.imshow(noisy_disp)
            ax.imshow(raw_wd, cmap='jet', alpha=0.5, vmin=v_min, vmax=v_max)
            ax.text(0.95, 0.05, f"{raw_wd.max():.2f}", color='white', fontweight='bold', transform=ax.transAxes, ha='right', bbox=dict(facecolor='black', alpha=0.5))
            ax.set_xticks([]); ax.set_yticks([])

        wd_label = ratio_suffix.replace('pct', '%').replace('_', '.')
        pc_label = "10%" if path_pc and "10pct" in path_pc else "100%"

        cols = ['Original', 'Noisy Input', f'PatchCore ({pc_label})', f'WEDGE-Net ({wd_label})']
        for ax, col in zip(axes[0], cols):
            ax.set_title(col, fontsize=18, fontweight='bold', pad=10)

        save_path = os.path.join(BASE_DIR, f"Figure_Noise_Robustness_{category}_{ratio_suffix}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {save_path}")

    print("\n[Done] All tasks complete.")

if __name__ == '__main__':
    run_noise_experiment()
