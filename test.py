import matplotlib
matplotlib.use('Agg') # Enable headless plotting
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
import numpy as np
import os
from torchvision import transforms

from model import WEDGE_Net
from dataset import MVTecDataset
import config 

# Define All Categories
ALL_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

def get_ratio_name(ratio):
    """Helper to match folder naming convention."""
    if ratio >= 0.99: return "100pct"
    elif abs(ratio - 0.1) < 1e-6: return "10pct"
    elif abs(ratio - 0.01) < 1e-6: return "1pct"
    elif abs(ratio - 0.001) < 1e-6: return "0_1pct"
    else: return f"{int(ratio*100)}pct"

def normalize_map(m):
    """Normalizes a feature map to [0, 1] range."""
    m_min = m.min()
    m_max = m.max()
    if m_max - m_min > 1e-5:
        return (m - m_min) / (m_max - m_min)
    return m

def test_inference():
    # =============================================================
    # [1] Configuration & Setup
    # =============================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = getattr(config, 'SAVE_DIR', 'WEDGE-Net')
    
    # 1. Determine Target Ratios
    raw_ratio = getattr(config, 'SAMPLING_RATIO', '0.1')
    if str(raw_ratio).lower() == 'all':
        # Default to 0.01 if 'all' is selected for visualization to prevent overflow
        print(" [Info] 'all' ratio selected. Defaulting to 1% (0.01) for visualization.")
        target_ratios = [0.01]
    else:
        target_ratios = [float(raw_ratio)]

    # 2. Determine Target Categories
    raw_cat = getattr(config, 'CATEGORY', 'all')
    if str(raw_cat).lower() == 'all':
        target_categories = ALL_CATEGORIES
    else:
        target_categories = [str(raw_cat)]

    # Settings for visualization limit
    MAX_IMAGES = 10  # Process only 10 images per category

    # =============================================================
    # [2] Main Loops (Ratio -> Category)
    # =============================================================
    for ratio in target_ratios:
        ratio_folder_name = get_ratio_name(ratio)
        print(f"\n" + "#"*60)
        print(f" Processing Ratio: {ratio_folder_name}")
        print(f"#"*60)

        for category in target_categories:
            print(f"\n >> Category: {category.upper()} ({ratio_folder_name})")

            # Path Setup
            model_filename = f"model_data_{category}_{ratio_folder_name}.pt"
            model_path = os.path.join(base_dir, ratio_folder_name, model_filename)
            save_dir = os.path.join(base_dir, ratio_folder_name, "results", category)
            
            # Check Model Existence
            if not os.path.exists(model_path):
                print(f"   [Skip] Model not found: {model_path}")
                continue

            # Load Model
            try:
                checkpoint = torch.load(model_path, map_location=device)
                if 'memory_bank' in checkpoint:
                    memory_bank = checkpoint['memory_bank'].to(device)
                else:
                    memory_bank = checkpoint.to(device)

                model = WEDGE_Net(use_semantic=config.USE_SEMANTIC).to(device)
                model.eval()
            except Exception as e:
                print(f"   [Error] Failed to load model: {e}")
                continue
            
            # Load Dataset
            try:
                test_dataset = MVTecDataset(root_dir=config.DATA_PATH, category=category, phase='test')
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            except Exception as e:
                print(f"   [Error] Failed to load dataset: {e}")
                continue

            os.makedirs(save_dir, exist_ok=True)

            inv_normalize = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )

            saved_count = 0 

            with torch.no_grad():
                for i, (image, label, mask, img_path) in enumerate(test_loader):
                    # [Condition 1] Stop if limit reached
                    if saved_count >= MAX_IMAGES:
                        break
                    
                    # [Condition 2] Skip Normal images (Label 0 = Good, 1 = Defect)
                    if label.item() == 0:
                        continue
                    
                    filename = os.path.basename(img_path[0])
                    image = image.to(device)
                    
                    # Forward Pass
                    features, maps = model(image) 
                    
                    # Anomaly Scoring
                    B, C, H, W = features.shape
                    features_flat = features.view(C, -1).permute(1, 0)
                    dists = torch.cdist(features_flat, memory_bank, p=2) 
                    min_dist, _ = torch.min(dists, dim=1)
                    
                    # Process Score Map
                    score_map = min_dist.view(H, W).unsqueeze(0).unsqueeze(0)
                    score_map = F.interpolate(score_map, size=image.shape[2:], mode='bilinear', align_corners=False)
                    score_map = score_map.squeeze().cpu().numpy()
                    score_map = gaussian_filter(score_map, sigma=4.0)
                    
                    # Visualization Prep
                    img_vis = inv_normalize(image.squeeze().cpu()).permute(1, 2, 0).numpy()
                    img_vis = np.clip(img_vis, 0, 1)
                    gt_vis = mask.squeeze().cpu().numpy()
                    
                    # Frequency Map Logic
                    if isinstance(maps, dict) and 'freq' in maps: freq_raw = maps['freq']
                    elif isinstance(maps, (list, tuple)) and len(maps) > 0: freq_raw = maps[0]
                    else: freq_raw = torch.zeros_like(torch.tensor(score_map))
                    
                    if isinstance(freq_raw, torch.Tensor):
                         if freq_raw.dim() == 3: freq_raw = freq_raw.unsqueeze(0)
                         if freq_raw.dim() == 2: freq_raw = freq_raw.unsqueeze(0).unsqueeze(0)
                         freq_vis = F.interpolate(freq_raw, size=image.shape[2:], mode='bilinear', align_corners=False)
                         freq_vis = freq_vis.cpu().squeeze()
                         if freq_vis.dim() == 3: freq_vis = freq_vis.mean(dim=0) if freq_vis.shape[0] > 1 else freq_vis.squeeze(0)
                         freq_vis = freq_vis.numpy()
                    else: freq_vis = freq_raw

                    # Semantic Map Logic
                    if isinstance(maps, dict) and 'sem' in maps: sem_raw = maps['sem']
                    elif isinstance(maps, (list, tuple)) and len(maps) > 1: sem_raw = maps[1]
                    else: sem_raw = torch.zeros_like(torch.tensor(score_map))

                    if isinstance(sem_raw, torch.Tensor):
                         if sem_raw.dim() == 3: sem_raw = sem_raw.unsqueeze(0)
                         if sem_raw.dim() == 2: sem_raw = sem_raw.unsqueeze(0).unsqueeze(0)
                         sem_vis = F.interpolate(sem_raw, size=image.shape[2:], mode='bilinear', align_corners=False)
                         sem_vis = sem_vis.cpu().squeeze()
                         if sem_vis.dim() == 3: sem_vis = sem_vis.mean(dim=0) if sem_vis.shape[0] > 1 else sem_vis.squeeze(0)
                         sem_vis = sem_vis.numpy()
                    else: sem_vis = sem_raw

                    # Normalize & Plot
                    score_vis = normalize_map(score_map)
                    freq_vis = normalize_map(freq_vis)
                    sem_vis = normalize_map(sem_vis)

                    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
                    axes[0].imshow(img_vis); axes[0].set_title("(a) Input"); axes[0].axis('off')
                    axes[1].imshow(gt_vis, cmap='gray'); axes[1].set_title("(b) GT"); axes[1].axis('off')
                    axes[2].imshow(freq_vis, cmap='gray'); axes[2].set_title("(c) Frequency Map"); axes[2].axis('off')
                    axes[3].imshow(sem_vis, cmap='magma'); axes[3].set_title("(d) Semantic Map"); axes[3].axis('off')
                    axes[4].imshow(score_vis, cmap='jet'); axes[4].set_title("(e) Anomaly Score"); axes[4].axis('off')
                    
                    axes[5].imshow(img_vis)
                    axes[5].imshow(score_vis, cmap='jet', alpha=0.5)
                    axes[5].set_title("(f) Result"); axes[5].axis('off')

                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, filename))
                    plt.close()
                    
                    saved_count += 1 # Increment counter

            print(f"   Saved {saved_count} defect samples to: {save_dir}")

    print("\n[Done] All visualization tasks complete.")

if __name__ == "__main__":
    test_inference()
