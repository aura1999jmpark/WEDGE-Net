import matplotlib
matplotlib.use('Agg') # Enable headless plotting for server environments
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
import numpy as np
import os
from torchvision import transforms

# Ensure correct imports based on your file structure

from model import WEDGE_Net
from dataset import MVTecDataset
import config 

def get_ratio_name(ratio):
    """Helper to match folder naming convention."""
    if ratio == 1.0: return "100pct"
    elif ratio == 0.1: return "10pct"
    elif ratio == 0.01: return "1pct"
    else: return f"{int(ratio*100)}pct"

def normalize_map(m):
    """Normalizes a feature map to [0, 1] range for visualization."""
    m_min = m.min()
    m_max = m.max()
    if m_max - m_min > 1e-5:
        return (m - m_min) / (m_max - m_min)
    return m

def test_inference():
    # =============================================================
    # [1] Configuration & Path Setup
    # =============================================================
    category = config.CATEGORY
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    base_dir = getattr(config, 'SAVE_DIR', 'WEDGE-Net')
    raw_ratio = getattr(config, 'SAMPLING_RATIO', '0.1')
    
    if raw_ratio == 'all':
        target_ratio = 0.1
    else:
        target_ratio = float(raw_ratio)
    
    ratio_folder_name = get_ratio_name(target_ratio)
    model_filename = f"model_data_{category}_{ratio_folder_name}.pt"
    model_path = os.path.join(base_dir, ratio_folder_name, model_filename)
    
    print(f"==================================================")
    print(f" [Test Setup] Generating 6-Column Visualization")
    print(f" - Category : {category}")
    print(f" - Model    : {model_path}")
    print(f"==================================================")

    # =============================================================
    # [2] Load Model
    # =============================================================
    if not os.path.exists(model_path):
        print(f"[Error] Model file not found at: {model_path}")
        return
        
    checkpoint = torch.load(model_path, map_location=device)
    memory_bank = checkpoint['memory_bank'].to(device)
    
    # Initialize Model
    model = WEDGE_Net(use_semantic=config.USE_SEMANTIC).to(device)
    model.eval()
    
    # =============================================================
    # [3] Data & Vis Setup
    # =============================================================
    test_dataset = MVTecDataset(root_dir=config.DATA_PATH, category=category, phase='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Save results in: z_test/10pct/results/tile
    save_dir = os.path.join(base_dir, ratio_folder_name, "results", category)
    os.makedirs(save_dir, exist_ok=True)
    
    # Standard ImageNet Denormalization
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    print(f"Saving 6-column figures to: {save_dir}")
    
    with torch.no_grad():
        for i, (image, label, mask, img_path) in enumerate(test_loader):
            # Process all images (remove 'if i >= 20: break' if you want all)
            # if i >= 20: break 
            
            filename = os.path.basename(img_path[0])
            image = image.to(device)
            
            # ---------------------------------------------------------
            # Forward Pass
            # ---------------------------------------------------------
            features, maps = model(image) 
            
            # 1. Anomaly Scoring (Euclidean Distance)
            B, C, H, W = features.shape
            features_flat = features.view(C, -1).permute(1, 0)
            dists = torch.cdist(features_flat, memory_bank, p=2) 
            min_dist, _ = torch.min(dists, dim=1)
            
            # 2. Process (e) Anomaly Score Map
            score_map = min_dist.view(H, W).unsqueeze(0).unsqueeze(0)
            score_map = F.interpolate(score_map, size=image.shape[2:], mode='bilinear', align_corners=False)
            score_map = score_map.squeeze().cpu().numpy()
            score_map = gaussian_filter(score_map, sigma=4.0) # Smoothing
            
            # 3. Process Maps for Visualization
            # (a) Input Image
            img_vis = inv_normalize(image.squeeze().cpu()).permute(1, 2, 0).numpy()
            img_vis = np.clip(img_vis, 0, 1)

            # (b) Ground Truth (GT)
            gt_vis = mask.squeeze().cpu().numpy()
            
            # =========================================================
            # [CRITICAL FIX] Safe Dimension Handling for Visualization
            # =========================================================
            
            # --- (c) Frequency Attention Map ---
            if isinstance(maps, dict) and 'freq' in maps:
                freq_raw = maps['freq']
            elif isinstance(maps, (list, tuple)) and len(maps) > 0:
                freq_raw = maps[0]
            else:
                freq_raw = torch.zeros_like(torch.tensor(score_map))
            
            if isinstance(freq_raw, torch.Tensor):
                 # Ensure 4D [B, C, H, W] for interpolate
                 if freq_raw.dim() == 3: freq_raw = freq_raw.unsqueeze(0)
                 if freq_raw.dim() == 2: freq_raw = freq_raw.unsqueeze(0).unsqueeze(0)
                 
                 freq_vis = F.interpolate(freq_raw, size=image.shape[2:], mode='bilinear', align_corners=False)
                 freq_vis = freq_vis.cpu().squeeze() # [C, H, W] or [H, W]
                 
                 # Only average if channels exist (Dim=3). If already 2D (Dim=2), do NOT average.
                 if freq_vis.dim() == 3:
                     if freq_vis.shape[0] == 1:
                         freq_vis = freq_vis.squeeze(0)
                     else:
                         freq_vis = freq_vis.mean(dim=0)
                 
                 freq_vis = freq_vis.numpy()
            else:
                freq_vis = freq_raw

            # --- (d) Semantic Attention Map ---
            if isinstance(maps, dict) and 'sem' in maps:
                sem_raw = maps['sem']
            elif isinstance(maps, (list, tuple)) and len(maps) > 1:
                sem_raw = maps[1]
            else:
                sem_raw = torch.zeros_like(torch.tensor(score_map))

            if isinstance(sem_raw, torch.Tensor):
                 # Ensure 4D [B, C, H, W] for interpolate
                 if sem_raw.dim() == 3: sem_raw = sem_raw.unsqueeze(0)
                 if sem_raw.dim() == 2: sem_raw = sem_raw.unsqueeze(0).unsqueeze(0)

                 sem_vis = F.interpolate(sem_raw, size=image.shape[2:], mode='bilinear', align_corners=False)
                 sem_vis = sem_vis.cpu().squeeze() # [C, H, W] or [H, W]
                 
                 # Only average if channels exist (Dim=3). If already 2D (Dim=2), do NOT average.
                 if sem_vis.dim() == 3:
                     if sem_vis.shape[0] == 1:
                         sem_vis = sem_vis.squeeze(0)
                     else:
                         sem_vis = sem_vis.mean(dim=0)
                     
                 sem_vis = sem_vis.numpy()
            else:
                sem_vis = sem_raw

            # Normalize maps for plotting
            score_vis = normalize_map(score_map)
            freq_vis = normalize_map(freq_vis)
            sem_vis = normalize_map(sem_vis)

            # ---------------------------------------------------------
            # Plotting: 6 Columns
            # ---------------------------------------------------------
            fig, axes = plt.subplots(1, 6, figsize=(24, 4))
            
            # (a) Input
            axes[0].imshow(img_vis)
            axes[0].set_title("(a) Input")
            axes[0].axis('off')

            # (b) GT
            axes[1].imshow(gt_vis, cmap='gray')
            axes[1].set_title("(b) GT")
            axes[1].axis('off')

            # (c) Frequency Map (Gray/Structural)
            axes[2].imshow(freq_vis, cmap='gray')
            axes[2].set_title("(c) Frequency Map")
            axes[2].axis('off')

            # (d) Semantic Map (Magma/Energy)
            axes[3].imshow(sem_vis, cmap='magma')
            axes[3].set_title("(d) Semantic Map")
            axes[3].axis('off')

            # (e) Anomaly Score Map (Jet Heatmap)
            axes[4].imshow(score_vis, cmap='jet')
            axes[4].set_title("(e) Anomaly Score")
            axes[4].axis('off')

            # (f) Localization Result (Overlay)
            axes[5].imshow(img_vis)
            axes[5].imshow(score_vis, cmap='jet', alpha=0.5) # Overlay
            axes[5].set_title("(f) Result")
            axes[5].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, filename))
            plt.close()

    print(f"Visualization Complete! Results saved to: {save_dir}")

if __name__ == "__main__":
    test_inference()