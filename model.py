import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import config

# [InputSpace_DWT remains unchanged as requested]
class InputSpace_DWT(nn.Module):
    def __init__(self): 
        super(InputSpace_DWT, self).__init__()
        self.to_gray = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.to_gray.weight.data = torch.tensor([[[[0.299]], [[0.587]], [[0.114]]]])
        self.to_gray.weight.requires_grad = False
        self.wavelet_type = getattr(config, 'WAVELET_TYPE', 'haar')
        if self.wavelet_type == 'bior2.2':
            self.register_buffer('kernel', self._get_bior22_kernel())
            self.pad_size = 2 
        else:
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
        x_gray = self.to_gray(x_input) 
        if self.pad_size > 0:
            x_gray = F.pad(x_gray, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), mode='reflect')
        dwt_maps = F.conv2d(x_gray, self.kernel, stride=1, padding=0)
        if self.wavelet_type == 'haar':
             dwt_maps = F.pad(dwt_maps, (0, 1, 0, 1)) 
        total_energy = torch.sum(dwt_maps ** 2, dim=1, keepdim=True)
        guidance_map = torch.sqrt(total_energy)
        guidance_map = (guidance_map - guidance_map.min()) / (guidance_map.max() - guidance_map.min() + 1e-6)
        return guidance_map

# ==========================================================================
# [V3_2] WEDGE-Net: Wavelet-Enhanced Dual-stream Guided Embedding Network
# ==========================================================================
class WEDGE_Net(nn.Module):
    def __init__(self, use_semantic=False):
        super(WEDGE_Net, self).__init__()
        
        self.use_semantic = use_semantic 
        
        # 1. Frequency Stream (Noise Filtering)
        # Filters out high-frequency environmental noise using DWT.
        self.input_dwt = InputSpace_DWT()
        
        # 2. Context Stream (Semantic Representation)
        # Uses a pre-trained ResNet-50 backbone to capture object semantics.
        # We explicitly use the standard ResNet-50 (not WideResNet) for efficiency.
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2 # Output: 512 channels
        self.layer3 = backbone.layer3 # Output: 1024 channels

        # 3. Frequency Attention Encoder
        # Generates a spatial attention map from the frequency guidance to modulate the backbone features.
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 512, kernel_size=3, padding=1), 
            nn.ReLU()
        )
        
        # Project frequency features to match Layer 3 dimensions (512 -> 1024)
        self.proj_to_f3 = nn.Conv2d(512, 1024, kernel_size=1, bias=False)
        
        # ==========================================================
        # Dimensionality Projection Layer (Optional)
        # ==========================================================
        # We calculate the total feature dimension based on the configuration.
        # Base: 512 (Layer 2) + 1024 (Layer 3) = 1536
        # With Semantic: 1536 + 512 = 2048
        current_dim = 1536
        if self.use_semantic:
            current_dim += 512 
            
        self.target_dim = 1024
        
        # [Configuration Control]
        # Set to False to maintain full feature resolution (2048-dim) for maximum accuracy.
        # Set to True to enable projection (1024-dim) for reduced memory footprint.
        self.use_projection = False 

        if self.use_projection:
            # Linear projection layer (Random Projection equivalent if bias=False)
            self.dim_reducer = nn.Conv2d(current_dim, self.target_dim, kernel_size=1, bias=False)
            print(f"[WEDGE-Net] Projection Enabled: Compressed features {current_dim}D -> {self.target_dim}D")
        else:
            # If disabled, we do not initialize the layer to save parameters.
            self.dim_reducer = None
            # print(f"[WEDGE-Net] Projection Disabled: Using full features ({current_dim}D)")


    def forward(self, x):
        # -----------------------------------------------------------
        # Stream 1: Frequency Guidance Generation
        # -----------------------------------------------------------
        guidance_map = self.input_dwt(x)
        
        # -----------------------------------------------------------
        # Stream 2: Semantic Feature Extraction (Context Stream)
        # -----------------------------------------------------------
        x_stem = self.stem(x)
        f2 = self.layer2(x_stem) 
        f3 = self.layer3(f2)      
        
        f3_up = F.interpolate(f3, size=(28, 28), mode='bilinear', align_corners=False)

        # -----------------------------------------------------------
        # Frequency-Aware Attention Gating
        # -----------------------------------------------------------
        # Modulate semantic features using the frequency guidance map
        guide_small = F.interpolate(guidance_map, size=(28, 28), mode='bilinear', align_corners=False)
        f_freq = self.freq_encoder(guide_small)

        # Apply attention to Layer 2 features
        f_freq_sigmoid = torch.sigmoid(f_freq) 
        f2_final = f2 * (1 + f_freq_sigmoid)
        
        # Apply attention to Layer 3 features
        f_freq_up = self.proj_to_f3(f_freq) 
        f_freq_sigmoid_up = torch.sigmoid(f_freq_up)
        f3_final = f3_up * (1 + f_freq_sigmoid_up)

        # -----------------------------------------------------------
        # Feature Aggregation & Semantic Guidance
        # -----------------------------------------------------------
        feat_to_concat = []
        feat_to_concat.extend([
            F.normalize(f2_final, p=2, dim=1),
            F.normalize(f3_final, p=2, dim=1)
        ])

        # Generate Semantic Attention Mask (Object Consistency)
        raw_source = f2 
        amp_vis = torch.mean(torch.pow(raw_source, 2), dim=1, keepdim=True)
        sem_vis = (amp_vis - amp_vis.min()) / (amp_vis.max() - amp_vis.min() + 1e-6)

        if self.use_semantic:
            # Use Layer 2 activation magnitude as a semantic mask
            amp_train = torch.mean(torch.abs(raw_source), dim=1, keepdim=True)
            sem_train = (amp_train - amp_train.min()) / (amp_train.max() - amp_train.min() + 1e-6)
            
            norm_raw = F.normalize(raw_source, p=2, dim=1)
            # Weighted concatenation of semantic features
            feat_to_concat.append(norm_raw * sem_train * 2.0)
        
        # Final Feature Vector Construction
        features = torch.cat(feat_to_concat, dim=1)
        
        # Optional Dimensionality Reduction
        if self.use_projection and self.dim_reducer is not None:
             features = self.dim_reducer(features)

        # Spatial Smoothing for robustness against local noise
        features = F.avg_pool2d(features, kernel_size=3, stride=1, padding=1)
        
        maps = {
            'freq': guidance_map,  
            'sem': sem_vis      
        }
        
        return features, maps
