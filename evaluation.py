import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
import os
import numpy as np
import csv
from tqdm import tqdm

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    print("Error: 'scikit-learn' is required. Install via: pip install scikit-learn")
    exit()

from model import WEDGE_Net
from dataset import MVTecDataset
import config

# Full MVTec AD category list
ALL_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

def get_target_model_path(base_dir, category, ratio_name):
    """
    Smart Path Finder:
    Searches for the model file in multiple possible locations.
    Priority 1: Sub-directory (e.g., save_dir/10pct/model_data_cat_10pct.pt)
    Priority 2: Root directory (e.g., save_dir/model_data_cat_10pct.pt)
    Priority 3: Legacy name (e.g., save_dir/model_data_cat.pt)
    """
    # 1. Check Sub-directory structure (Standard for 'all' training)
    path_sub = os.path.join(base_dir, ratio_name, f"model_data_{category}_{ratio_name}.pt")
    if os.path.exists(path_sub):
        return path_sub

    # 2. Check Root directory structure (Standard for single training)
    path_root = os.path.join(base_dir, f"model_data_{category}_{ratio_name}.pt")
    if os.path.exists(path_root):
        return path_root
        
    # 3. Check Legacy/Simple naming (Only if specific suffix fails)
    # This acts as a fallback if the file was saved without suffix
    path_legacy = os.path.join(base_dir, f"model_data_{category}.pt")
    if os.path.exists(path_legacy):
        # Caution: We only trust this if we are looking for the configured ratio or 100%
        return path_legacy

    return None

def evaluate_category(category, device, save_dir):
    """
    Evaluates a specific category. 
    It iterates through 100%, 10%, 1%, and 0.1% (0_1pct).
    """
    # [Update] Added 0.1% (0.001) support
    target_ratios = [1.0, 0.1, 0.01, 0.001]
    ratio_names = {
        1.0: "100pct", 
        0.1: "10pct", 
        0.01: "1pct", 
        0.001: "0_1pct"
    }
    
    cat_result = {'Category': category}
    
    # Setup Model (Architecture is shared)
    model = WEDGE_Net(use_semantic=config.USE_SEMANTIC).to(device)
    model.eval()

    # Setup DataLoader (Loaded once per category)
    try:
        test_dataset = MVTecDataset(root_dir=config.DATA_PATH, category=category, phase='test')
        test_loader = DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=getattr(config, 'NUM_WORKERS', 4)
        )
    except Exception as e:
        print(f"[Error] Failed to load dataset for {category}: {e}")
        return None

    print(f"\n>>> Category: {category.upper()}")

    # Loop through ratios to create a comprehensive row
    for ratio in target_ratios:
        r_name = ratio_names[ratio]
        
        # Determine display key (e.g., "0.1%")
        if ratio == 0.001: display_key = "0.1%"
        elif ratio == 0.01: display_key = "1%"
        elif ratio == 0.1: display_key = "10%"
        else: display_key = "100%"

        # [SMART FINDER] Look for the file
        model_path = get_target_model_path(save_dir, category, r_name)
        
        if model_path is None:
            # If file doesn't exist in any expected location
            # print(f"    [Skip] Model for {r_name} not found.")
            cat_result[display_key] = "N/A"
            continue

        # Load Memory Bank
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'memory_bank' in checkpoint:
                memory_bank = checkpoint['memory_bank'].float()
            else:
                memory_bank = checkpoint.float()
        except Exception as e:
            print(f"    [Error] Failed to load {model_path}: {e}")
            cat_result[display_key] = "Error"
            continue

        # Inference Loop
        image_scores = []
        gt_labels = []

        with torch.no_grad():
            for image, label, _, _ in test_loader:
                image = image.to(device)
                features, _ = model(image)

                B, C, H, W = features.shape
                features_flat = features.view(C, -1).permute(1, 0).float()
                
                dists = torch.cdist(features_flat, memory_bank, p=2)
                min_dist, _ = torch.min(dists, dim=1)
                
                score_map = min_dist.view(H, W).unsqueeze(0).unsqueeze(0)
                score_map = F.interpolate(
                    score_map, 
                    size=(config.IMAGE_SIZE, config.IMAGE_SIZE), 
                    mode='bilinear', 
                    align_corners=False
                )
                score_map = score_map.squeeze().cpu().numpy()
                score_map = gaussian_filter(score_map, sigma=4.0)
                
                image_scores.append(score_map.max())
                gt_labels.append(label.item())

        # Compute AUROC
        try:
            if len(set(gt_labels)) < 2:
                cat_result[display_key] = 0.0
            else:
                auroc = roc_auc_score(gt_labels, image_scores) * 100
                cat_result[display_key] = round(auroc, 2)
                print(f"    [{display_key}] AUROC: {auroc:.2f}%  (File: ...{os.path.basename(model_path)})")
        except ValueError:
            cat_result[display_key] = 0.0

    return cat_result

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    device = torch.device('cuda')

    # 1. Determine Target Categories
    raw_cat = getattr(config, 'CATEGORY', 'all')
    if str(raw_cat).lower() == 'all':
        print("Mode: Processing ALL categories.")
        target_categories = ALL_CATEGORIES
    elif ',' in str(raw_cat):
        target_categories = [cat.strip() for cat in raw_cat.split(',')]
        print(f"Mode: Processing specific list -> {target_categories}")
    else:
        print(f"Mode: Single category processing ({raw_cat}).")
        target_categories = [str(raw_cat)]

    # 2. Setup Save Directory (From Config)
    save_dir = getattr(config, 'SAVE_DIR', 'WEDGE-Net')
    if not os.path.exists(save_dir):
        print(f"Error: Directory {save_dir} does not exist. Please check config or run training.")
        return

    print(f"Source Directory: {save_dir}")

    # 3. Run Evaluation
    final_results = []
    
    # [Update] Added 0.1% to CSV Header
    fieldnames = ['Category', '100%', '10%', '1%', '0.1%']
    
    for cat in target_categories:
        res = evaluate_category(cat, device, save_dir)
        if res is not None:
            final_results.append(res)

    # 4. Save CSV (Always in the Root SAVE_DIR)
    if final_results:
        # Calculate Average
        avg_row = {'Category': 'AVERAGE'}
        
        # [Update] Loop through all ratio columns including 0.1%
        for key in ['100%', '10%', '1%', '0.1%']:
            valid_scores = [r[key] for r in final_results if key in r and isinstance(r[key], (int, float))]
            if valid_scores:
                avg_row[key] = round(sum(valid_scores) / len(valid_scores), 2)
            else:
                avg_row[key] = "N/A"
        final_results.append(avg_row)

        csv_path = os.path.join(save_dir, "final_evaluation_results.csv")
        
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_results)

        print(f"\n[Done] Evaluation Results saved to: {csv_path}")
        print(f"      Average -> 100%: {avg_row['100%']} | 10%: {avg_row['10%']} | 1%: {avg_row['1%']} | 0.1%: {avg_row['0.1%']}")

if __name__ == "__main__":
    main()
