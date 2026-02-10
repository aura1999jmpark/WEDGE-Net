"""
[Benchmark Script] WEDGE-Net Inference Speed Measurement

This script measures the exact inference FPS (Frames Per Second) of WEDGE-Net models.
It is designed to validate the "Extreme Efficiency" claims made in the paper.

Methodology:
1. Environment: Single NVIDIA GPU (e.g., RTX 4090) or CPU.
2. Input: Single image batch (B=1) to simulate real-time edge processing latency.
3. Process: 
   - Feature Extraction (Backbone + Frequency/Semantic Modules)
   - Anomaly Scoring (Distance calculation against Memory Bank)
4. Protocol:
   - Warm-up phase (10 iterations) to stabilize GPU clock.
   - Synchronization (torch.cuda.synchronize) before and after timing.
   - Average over 200 iterations for statistical stability.

Note on Comparison:
- Baseline (PatchCore) speeds reported in the paper were measured using the 
  official PatchCore implementation under identical hardware conditions.
"""

import torch
import time
import numpy as np
import os
import glob
import csv
from model import WEDGE_Net
import config

def measure_fps_exact():
    # =========================================================
    # 1. Setup & Configuration
    # =========================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # [Target Path]
    base_dir = getattr(config, 'SAVE_DIR', 'WEDGE-Net')
    
    # [Logic] Determine Suffix based on Config Ratio
    raw_ratio = getattr(config, 'SAMPLING_RATIO', 'all')
    target_suffix = ""
    
    # Convert ratio to suffix string for Folder Search & Filename
    if str(raw_ratio) == '0.001': target_suffix = "0_1pct"
    elif str(raw_ratio) == '0.01': target_suffix = "1pct"
    elif str(raw_ratio) == '0.1': target_suffix = "10pct"
    elif str(raw_ratio) == '1.0': target_suffix = "100pct"
    else: target_suffix = "all" # If 'all' or specific number not in list

    # [Mod] Output CSV Path -> Append suffix to filename
    # Example: benchmark_fps_results_1pct.csv
    csv_filename = f"benchmark_fps_results_{target_suffix}.csv"
    output_csv = os.path.join(base_dir, csv_filename)

    # Path Searching Logic
    if target_suffix and target_suffix != "all":
        pt_folder = os.path.join(base_dir, target_suffix)
        print(f"🎯 Config Target Ratio: {raw_ratio} -> Looking into '{pt_folder}'")
    else:
        pt_folder = base_dir
        print(f"🎯 Config Target Ratio: {raw_ratio} -> Looking into '{pt_folder}' (Recursive)")

    print(f"🚀 [FPS Benchmark] Device: {device}")
    
    # Find .pt files
    if not os.path.exists(pt_folder):
        print(f"❌ Error: Folder '{pt_folder}' does not exist.")
        return

    pt_files = glob.glob(os.path.join(pt_folder, "**/*.pt"), recursive=True)
    
    # Filter: If specific suffix needed
    if target_suffix and target_suffix != "all":
        pt_files = [f for f in pt_files if target_suffix in f]

    if not pt_files:
        print(f"❌ Error: No matching .pt files found in '{pt_folder}'.")
        return

    # =========================================================
    # 2. Initialize Model (Skeleton)
    # =========================================================
    print("⚙️ Initializing Model Backbone...")
    # Initialize with default; will be auto-configured later based on feature dim
    model = WEDGE_Net(use_semantic=config.USE_SEMANTIC).to(device)
    model.eval()
    
    # Dummy Input: Simulates a single image (Batch=1) for real-time latency test
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # =========================================================
    # 3. Start Benchmarking
    # =========================================================
    os.makedirs(base_dir, exist_ok=True)

    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['Filename', 'Category', 'Mode(Auto)', 'Ratio_Label', 'Memory_Size', 'FPS', 'Avg_Time(ms)']
        writer.writerow(header)
        
        print(f"📝 Saving results to: {output_csv}\n")

        for idx, file_path in enumerate(pt_files):
            filename = os.path.basename(file_path)
            
            # Parse Filename
            parts = filename.replace(".pt", "").split('_')
            category = parts[2] if len(parts) > 2 else "Unknown"
            ratio_label = parts[-1] if len(parts) > 3 else "Unknown"
            
            print(f"[{idx+1}/{len(pt_files)}] 📦 Analyzing: {filename} ({ratio_label})")
            
            try:
                checkpoint = torch.load(file_path, map_location=device)
                if isinstance(checkpoint, dict) and 'memory_bank' in checkpoint:
                    full_bank = checkpoint['memory_bank']
                else:
                    full_bank = checkpoint # Handle legacy format if any
                    
                current_bank = full_bank.to(device)
                
                total_samples = current_bank.shape[0]
                feature_dim = current_bank.shape[1] 
                
            except Exception as e:
                print(f"   ❌ Load Failed: {e}")
                continue

            # --- Auto-Configure Semantic Mode ---
            # Ensure the model architecture matches the saved feature dimension
            if feature_dim == 2048:
                model.use_semantic = True
                mode_str = "Sem_ON (2048)"
            elif feature_dim == 1536:
                model.use_semantic = False
                mode_str = "Sem_OFF (1536)"
            else:
                mode_str = f"Unknown ({feature_dim})"
                print(f"   ⚠️ Warning: Unexpected feature dimension {feature_dim}")
            
            # Update Model State
            # Perform a dummy pass to initialize any lazy-loaded layers
            with torch.no_grad():
                _ = model(dummy_input)

            # --- FPS Measurement ---
            
            # 1. Warm-up Phase
            # Runs the model multiple times to stabilize GPU clock frequencies/caches
            with torch.no_grad():
                for _ in range(10):
                    feat, _ = model(dummy_input)
                    feat_flat = feat.view(feat.shape[1], -1).permute(1, 0)
                    _ = torch.cdist(feat_flat, current_bank)

            # 2. Measurement Phase
            iterations = 200
            
            # [CRITICAL] Synchronize before starting timer for accurate GPU timing
            if device.type == 'cuda': torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(iterations):
                    # Step 1: Feature Extraction
                    features, _ = model(dummy_input)
                    B, C, H, W = features.shape
                    
                    # Step 2: Anomaly Scoring (Nearest Neighbor Search)
                    features_flat = features.view(C, -1).permute(1, 0)
                    dists = torch.cdist(features_flat, current_bank, p=2)
                    _, _ = torch.min(dists, dim=1)
            
            # [CRITICAL] Synchronize after finishing loops
            if device.type == 'cuda': torch.cuda.synchronize()
            total_time = time.time() - start_time
            
            fps = iterations / total_time
            avg_ms = (total_time / iterations) * 1000

            print(f"   👉 [{mode_str}] Memory: {total_samples} ({ratio_label}) | {fps:>7.2f} FPS")
            
            writer.writerow([filename, category, mode_str, ratio_label, total_samples, f"{fps:.2f}", f"{avg_ms:.2f}"])
            f.flush()

            del current_bank, checkpoint
            if device.type == 'cuda': torch.cuda.empty_cache()
            print("-" * 50)

    print(f"\n✅ Benchmark Complete! Results saved to '{output_csv}'.")

if __name__ == "__main__":
    measure_fps_exact()
