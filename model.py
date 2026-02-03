import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import config

# ==========================================================================
# [Module] Input-Space DWT (Unified Frequency Extractor)
# ==========================================================================
class InputSpace_DWT(nn.Module):
    """
    Performs discrete wavelet transform (DWT).
    Calculates Total Spectral Energy (LL included) to match Equation (2) in the paper.
    """
    def __init__(self): 
        super(InputSpace_DWT, self).__init__()
        
        # 1. Grayscale conversion using ITU-R 601-2 luma transform
        self.to_gray = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.to_gray.weight.data = torch.tensor([[[[0.299]], [[0.587]], [[0.114]]]])
        self.to_gray.weight.requires_grad = False
        
        # 2. Load wavelet kernel type from config
        self.wavelet_type = getattr(config, 'WAVELET_TYPE', 'haar')

        if self.wavelet_type == 'bior2.2':
            # Biorthogonal 2.2 (5x5 Kernel)
            self.register_buffer('kernel', self._get_bior22_kernel())
            self.pad_size = 2 
        else:
            # Standard Haar Wavelet (2x2 Kernel)
            self.register_buffer('kernel', self._get_haar_kernel())
            self.pad_size = 0 

    def _get_haar_kernel(self):
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        return torch.stack([ll, lh, hl, hh]).unsqueeze(1)
    
    def _get_bior22_kernel(self):
        lo_1d = torch.tensor([-0.125, 0.25, 0.75, 0.25, -0.125])
        hi_1d = torch.tensor([0.0, -0.5, 1.0, -0.5, 0.0]) 
        ll = torch.ger(lo_1d, lo_1d)
        lh = torch.ger(lo_1d, hi_1d)
        hl = torch.ger(hi_1d, lo_1d)
        hh = torch.ger(hi_1d, hi_1d)
        return torch.stack([ll, lh, hl, hh]).unsqueeze(1)

    def forward(self, x_input):
        # Convert to Grayscale
        x_gray = self.to_gray(x_input) 
        
        # Perform DWT Convolution
        if self.pad_size > 0:
            x_gray = F.pad(x_gray, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), mode='reflect')

        dwt_maps = F.conv2d(x_gray, self.kernel, stride=1, padding=0)
        
        # For Haar, align dimensions if necessary
        if self.wavelet_type == 'haar':
             dwt_maps = F.pad(dwt_maps, (0, 1, 0, 1)) 
        
        # Total Spectral Energy (Includes LL) 
        # Matches Equation: G = sqrt(LL^2 + LH^2 + HL^2 + HH^2)
        total_energy = torch.sum(dwt_maps ** 2, dim=1, keepdim=True)
        guidance_map = torch.sqrt(total_energy)
        
        # Min-Max Normalization for stable attention gating
        guidance_map = (guidance_map - guidance_map.min()) / (guidance_map.max() - guidance_map.min() + 1e-6)
        
        return guidance_map

# ==========================================================================
# [V 3_2] WEDGE_Net (Unified Architecture)
# ==========================================================================
class WEDGE_Net(nn.Module):
    def __init__(self, use_semantic=False):
        super(WEDGE_Net, self).__init__()
        
        self.use_semantic = use_semantic 
        
        # 1. Unified DWT Module
        self.input_dwt = InputSpace_DWT()
        
        # 2. Context Stream (Backbone: Pre-trained ResNet50)
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2 # Output: 512ch (Stride 8)
        self.layer3 = backbone.layer3 # Output: 1024ch (Stride 16)

        # 3. Side-Path: Frequency Encoder
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 512, kernel_size=3, padding=1), 
            nn.ReLU()
        )
        
        # Projection layer for channel matching (512 -> 1024 for Layer3)
        self.proj_to_f3 = nn.Conv2d(512, 1024, kernel_size=1, bias=False)
        
        print(f"[WEDGE-Net] Initialized. Wavelet: {self.input_dwt.wavelet_type.upper()}")

    def forward(self, x):
        """
        Forward pass with Wavelet-based Attention Gating.
        Returns features for training and maps for visualization.
        """
        # 1. Frequency Stream (Guidance Map)
        guidance_map = self.input_dwt(x)
        
        # 2. Context Stream
        x_stem = self.stem(x)
        f2 = self.layer2(x_stem) # [B, 512, 28, 28]
        f3 = self.layer3(f2)     # [B, 1024, 14, 14]
        
        # Upsample f3 to match f2 for concatenation later
        f3_up = F.interpolate(f3, size=(28, 28), mode='bilinear', align_corners=False)

        # 3. Side Path (Frequency Gating Mechanism)
        guide_small = F.interpolate(guidance_map, size=(28, 28), mode='bilinear', align_corners=False)
        f_freq = self.freq_encoder(guide_small)

        # 4. Multi-scale Feature Fusion via Attention Gating
        # Gating for Layer 2
        f_freq_sigmoid = torch.sigmoid(f_freq) 
        f2_final = f2 * (1 + f_freq_sigmoid)
        
        # Gating for Layer 3 (Projected)
        f_freq_up = self.proj_to_f3(f_freq) 
        f_freq_sigmoid_up = torch.sigmoid(f_freq_up)
        f3_final = f3_up * (1 + f_freq_sigmoid_up)

        # 5. Feature Aggregation
        feat_to_concat = []
        feat_to_concat.extend([
            F.normalize(f2_final, p=2, dim=1),
            F.normalize(f3_final, p=2, dim=1)
        ])

        # ==========================================================
        # [Split Logic] Training Map vs Visualization Map
        # ==========================================================
        raw_source = f2 
        
        # (A) Map for Training (L1-Norm based)
        # Using L1-Norm (Absolute value) ensures stable gradient flow during training.
        amp_train = torch.mean(torch.abs(raw_source), dim=1, keepdim=True)
        sem_train = (amp_train - amp_train.min()) / (amp_train.max() - amp_train.min() + 1e-6)

        # (B) Map for Visualization (L2-Norm based)
        # Using L2-Norm (Squared value) represents the spectral energy density,
        # providing a clearer representation of structural activations.
        amp_vis = torch.mean(torch.pow(raw_source, 2), dim=1, keepdim=True)
        sem_vis = (amp_vis - amp_vis.min()) / (amp_vis.max() - amp_vis.min() + 1e-6)

        # Optional: Semantic Module (Gating)
        if self.use_semantic:
            norm_raw = F.normalize(raw_source, p=2, dim=1)
            # IMPORTANT: Use 'sem_train' here to maintain training stability
            feat_to_concat.append(norm_raw * sem_train * 2.0)
        
        # Final Concatenation and Smoothing
        features = torch.cat(feat_to_concat, dim=1)
        features = F.avg_pool2d(features, kernel_size=3, stride=1, padding=1)
        
        # Return features (for Coreset) and maps dictionary (for Visualization)
        # Note: We return 'sem_vis' to generate clearer figures.
        maps = {
            'freq': guidance_map,  
            'sem': sem_vis    
        }
        
        return features, maps