import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import config

# ==========================================================================
# [Dataset] MVTec AD Data Loader
# ==========================================================================
class MVTecDataset(Dataset):
    """
    MVTec Anomaly Detection Dataset Loader.

    This class handles the loading and preprocessing of image samples and ground truth masks.
    It supports both the 'train' phase (nominal samples only) and the 'test' phase 
    (containing both nominal and anomalous samples with pixel-level annotations).

    Args:
        root_dir (str): Root directory path of the MVTec AD dataset.
        category (str): Target object category (e.g., 'bottle', 'transistor').
        phase (str): Operational phase, either 'train' or 'test'.
        img_size (int): Target resolution for resizing (default: config.IMAGE_SIZE).
        transform (callable, optional): Custom transformation pipeline. 
                                        If None, the default preprocessing pipeline is applied.
    """
    def __init__(self, root_dir, category, phase='train', img_size=config.IMAGE_SIZE, transform=None):
        self.root_dir = root_dir
        self.category = category
        self.phase = phase
        self.img_size = img_size
        
        # Define directory paths for images and ground truth masks
        self.img_dir = os.path.join(root_dir, category, phase)
        self.gt_dir = os.path.join(root_dir, category, 'ground_truth')
        
        self.data = []
        
        # Data Aggregation: Iterate through sub-directories to collect sample paths
        if os.path.exists(self.img_dir):
            types = sorted(os.listdir(self.img_dir))
            for t in types:
                t_dir = os.path.join(self.img_dir, t)
                if not os.path.isdir(t_dir): continue
                
                for f in sorted(os.listdir(t_dir)):
                    if f.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(t_dir, f)
                        
                        # Assign Binary Label: 0 for Normal (Good), 1 for Anomaly
                        label = 0 if t == 'good' else 1
                        
                        # Load Ground Truth Mask path for anomalous samples in the test set
                        mask_path = None
                        if phase == 'test' and label == 1:
                            mask_path = os.path.join(self.gt_dir, t, f.replace('.png', '_mask.png'))
                            
                        self.data.append((img_path, label, mask_path))
        
        # ----------------------------------------------------------------------
        # [Preprocessing Pipeline Configuration]
        # ----------------------------------------------------------------------
        # If an external transform is provided (e.g., for specific PatchCore generation), use it.
        # Otherwise, initialize the default preprocessing pipeline.
        if transform is not None:
            self.transform = transform
        else:
            # Default Pipeline:
            # 1. Resize the short edge to 256 to maintain aspect ratio.
            # 2. CenterCrop(224) to focus on the object, consistent with standard backbones.
            # 3. Normalize using ImageNet statistics.
            self.transform = transforms.Compose([
                transforms.Resize(256), 
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        
        # Ground Truth Mask Transformation
        # Resizes mask to matching dimensions without normalization (Nearest Neighbor interpolation).
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        Returns:
            img (Tensor): Preprocessed image tensor.
            label (int): Binary classification label (0: Normal, 1: Anomaly).
            mask (Tensor): Pixel-level ground truth mask (Zero tensor for normal samples).
            img_path (str): File path of the source image.
        """
        img_path, label, mask_path = self.data[idx]
        
        # Load Image and Apply Transforms
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        # Load Mask if available
        if self.phase == 'test' and mask_path and os.path.exists(mask_path):
            mask = self.mask_transform(Image.open(mask_path).convert('L'))
        else:
            # Return a zero-filled tensor for normal samples or missing masks
            mask = torch.zeros(1, self.img_size, self.img_size)
            
        return img, label, mask, img_path
