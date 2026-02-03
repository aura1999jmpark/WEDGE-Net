import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import config
# ==========================================================================
# [Dataset] MVTec AD Dataset Loader
# ==========================================================================
class MVTecDataset(Dataset):
    """
    Standard PyTorch Dataset for MVTec AD.
    Loads images, labels, and ground truth masks for anomaly detection.
    
    Args:
        root_dir (str): Path to the MVTec AD dataset root.
        category (str): Target category (e.g., 'bottle', 'tile').
        phase (str): 'train' or 'test'.
        img_size (int): Target image resolution (default: from config).
    """
    def __init__(self, root_dir, category, phase='train', img_size=config.IMAGE_SIZE):
        self.root_dir = root_dir
        self.category = category
        self.phase = phase
        self.img_size = img_size
        
        # Define paths for images and ground truth
        self.img_dir = os.path.join(root_dir, category, phase)
        self.gt_dir = os.path.join(root_dir, category, 'ground_truth')
        
        self.data = []
        
        # Load file paths
        # 'train' phase usually contains only 'good' samples.
        # 'test' phase contains 'good' and various defect types.
        if os.path.exists(self.img_dir):
            types = sorted(os.listdir(self.img_dir))
            for t in types:
                t_dir = os.path.join(self.img_dir, t)
                if not os.path.isdir(t_dir): continue
                
                for f in sorted(os.listdir(t_dir)):
                    if f.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(t_dir, f)
                        
                        # Label: 0 for normal (good), 1 for anomaly
                        label = 0 if t == 'good' else 1
                        
                        # Load Ground Truth Mask for anomalous test samples
                        mask_path = None
                        if phase == 'test' and label == 1:
                            mask_path = os.path.join(self.gt_dir, t, f.replace('.png', '_mask.png'))
                            
                        self.data.append((img_path, label, mask_path))
        
        # Standard ImageNet Normalization
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Mask Transformation (No normalization needed)
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label, mask_path = self.data[idx]
        
        # Load Image
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        # Load Mask
        if self.phase == 'test' and mask_path and os.path.exists(mask_path):
            mask = self.mask_transform(Image.open(mask_path).convert('L'))
        else:
            # Return empty mask for normal samples or if GT is missing
            mask = torch.zeros(1, self.img_size, self.img_size)
            
        return img, label, mask, img_path