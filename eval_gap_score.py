import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random
import numpy as np

# Ensure correct imports
# Note: Ensure your model file is named 'model.py' or update this import accordingly.
from model import WEDGE_Net
from dataset import MVTecDataset
import config

# Full MVTec AD category list
ALL_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

def k_center_greedy(matrix, sampling_ratio, device):
    """
    Selects a subset of features using the K-Center Greedy algorithm (Coreset).
    Includes a progress bar for visualization.
    """
    num_samples = matrix.shape[0]
    num_to_select = int(num_samples * sampling_ratio)
    
    # If sampling ratio is effectively 1.0, return all indices
    if num_to_select >= num_samples:
        return torch.arange(num_samples)

    matrix = matrix.to(device).float()
    selected_indices = [0]
    
    # Initial distance calculation
    dists = torch.cdist(matrix[0].unsqueeze(0), matrix, p=2).squeeze(0)
    
    # Iteratively select the furthest point (Greedy selection)
    # Wrapped with tqdm to show progress
    for _ in tqdm(range(num_to_select - 1), desc=f"Sampling {int(sampling_ratio*100)}%"):
        next_idx = torch.argmax(dists).item()
        selected_indices.append(next_idx)
        
        new_dists = torch.cdist(matrix[next_idx].unsqueeze(0), matrix, p=2).squeeze(0)
        dists = torch.minimum(dists, new_dists)
        
    return torch.tensor(selected_indices).cpu()

def get_ratio_name(ratio):
    """
    Helper to convert float ratio to string suffix/folder name.
    1.0 -> '100pct', 0.1 -> '10pct', 0.01 -> '1pct'
    """
    if ratio == 1.0: return "100pct"
    elif ratio == 0.1: return "10pct"
    elif ratio == 0.01: return "1pct"
    else: return f"{int(ratio*100)}pct"

def train_category(category, device):
    """
    Constructs the memory bank for a specific category.
    Handles 'all' sampling ratios efficiently by extracting features once.
    """
    
    # 1. Determine Sampling Ratios to Process
    # [Modified] Logic updated to ALWAYS include 1.0 (100pct) regardless of config setting.
    raw_ratio = getattr(config, 'SAMPLING_RATIO', 1.0)
    
    if raw_ratio == 'all':
        target_ratios = [1.0, 0.1, 0.01]
        print(f"\n[Training] Category: {category} | Mode: Process ALL ratios (100%, 10%, 1%)")
    else:
        requested_ratio = float(raw_ratio)
        if requested_ratio == 1.0:
            target_ratios = [1.0]
        else:
            # If user requests 0.1, we process [1.0, 0.1] to ensure baseline exists
            # Sorted descending to ensure 100pct (1.0) is processed/saved first
            target_ratios = sorted(list(set([1.0, requested_ratio])), reverse=True)
            
        print(f"\n[Training] Category: {category} | Mode: 100pct + Target ratio ({requested_ratio})")

    # 2. Check if all target files already exist (Skip Logic)
    base_save_dir = getattr(config, 'SAVE_DIR', 'WEDGE-Net')
    all_exist = True
    for r in target_ratios:
        name = get_ratio_name(r)
        path = os.path.join(base_save_dir, name, f"model_data_{category}_{name}.pt")
        if not os.path.exists(path):
            all_exist = False
            break
    
    if all_exist:
        print(f"[Skip] All target files for '{category}' already exist.")
        return

    # 3. Model & Data Initialization
    train_dataset = MVTecDataset(root_dir=config.DATA_PATH, category=category, phase='train')
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS
    )
    
    model = WEDGE_Net(use_semantic=config.USE_SEMANTIC).to(device)
    model.eval() 
    
    # 4. Feature Extraction (Performed ONCE)
    features_list = []
    print(f"[Info] Extracting features from {len(train_dataset)} normal samples...")
    
    with torch.no_grad():
        for image, _, _, _ in tqdm(train_loader, desc=f"Extracting {category}"):
            image = image.to(device)
            features, _ = model(image) 
            
            # Flatten features: (B, C, H, W) -> (B*H*W, C)
            B, C, H, W = features.shape
            features_flat = features.view(B, C, -1).permute(0, 2, 1).reshape(-1, C)
            features_list.append(features_flat.cpu())
            
    full_features = torch.cat(features_list, dim=0)
    
    # Compute Statistics once
    feature_mean = full_features.mean(dim=0)
    feature_std = full_features.std(dim=0)
    
    # 5. Iterative Sampling and Saving
    sampling_method = getattr(config, 'SAMPLING_METHOD', 'coreset')

    for ratio in target_ratios:
        ratio_name = get_ratio_name(ratio)
        
        # Define Save Path: checkpoints/10pct/model_data_bottle_10pct.pt
        save_subdir = os.path.join(base_save_dir, ratio_name)
        os.makedirs(save_subdir, exist_ok=True)
        save_path = os.path.join(save_subdir, f"model_data_{category}_{ratio_name}.pt")
        
        if os.path.exists(save_path):
            print(f"  -> [Skip] {ratio_name} already exists.")
            continue

        # Perform Sampling
        if ratio >= 1.0:
            memory_bank = full_features
            print(f"  -> [Save] Saving Full Memory ({ratio_name})...")
        else:
            print(f"  -> [Sampling] Generating {ratio_name} coreset...")
            if sampling_method == 'coreset':
                # k_center_greedy includes a progress bar
                indices = k_center_greedy(full_features, ratio, device)
                memory_bank = full_features[indices]
            elif sampling_method == 'random':
                num_samples = full_features.shape[0]
                target_size = int(num_samples * ratio)
                indices = torch.randperm(num_samples)[:target_size]
                memory_bank = full_features[indices]
            else:
                raise ValueError(f"Unknown method: {sampling_method}")

        # Save
        torch.save({
            'memory_bank': memory_bank,
            'feature_mean': feature_mean,
            'feature_std': feature_std
        }, save_path)
        
        print(f"      Saved to: {save_path} (Size: {memory_bank.shape})")

    print(f"[Done] Category {category} processing complete.\n")

def main():
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Determine target categories
    raw_cat = getattr(config, 'CATEGORY', 'all')
    
    if raw_cat.lower() == 'all':
        print("Mode: Processing ALL categories.")
        target_categories = ALL_CATEGORIES
    elif ',' in raw_cat:
        target_categories = [cat.strip() for cat in raw_cat.split(',')]
        print(f"Mode: Processing specific list -> {target_categories}")
    else:
        print(f"Mode: Single category processing ({raw_cat}).")
        target_categories = [raw_cat]

    # Run Loop
    for cat in target_categories:
        train_category(cat, device)

if __name__ == "__main__":
    main()
