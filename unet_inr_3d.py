import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================================================================
# 1. SIREN LAYER (Same as 2D - works with any dimensionality)
# =============================================================================

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.linear.in_features,
                     1 / self.linear.in_features
                )
            else:
                bound = math.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


# =============================================================================
# 2. INR HEAD - NOW HANDLES 3D COORDINATES
# =============================================================================

class INRHead3D(nn.Module):
    """
    3D INR that takes:
    - Normalized voxel coordinates (x, y, z) in [-1, 1]
    - Features from the LAST 3D U-Net decoder output
    """
    def __init__(
        self,
        feature_dim,      # Channels from final 3D U-Net output
        coord_dim=3,      # (x, y, z) coordinates - CHANGED FROM 2!
        hidden_dim=128,
        num_layers=3,
        omega_0=30.0,
        num_classes=3
    ):
        super().__init__()
        
        # Input: 3D coordinates + features from final decoder
        input_dim = coord_dim + feature_dim
        
        # Build SIREN network (same architecture, different input dim)
        layers = []
        layers.append(
            SineLayer(
                in_features=input_dim,
                out_features=hidden_dim,
                omega_0=omega_0,
                is_first=True
            )
        )
        
        for _ in range(num_layers - 1):
            layers.append(
                SineLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    omega_0=omega_0
                )
            )
        
        self.net = nn.Sequential(*layers)
        self.final = nn.Linear(hidden_dim, num_classes)

    def forward(self, coords, features):
        """
        coords:   (B, N, 3) - normalized voxel coordinates (x, y, z)
        features: (B, N, C) - features from final 3D U-Net output
        Returns:  (B, N, num_classes) - logits
        """
        # Concatenate 3D coordinates with features
        x = torch.cat([coords, features], dim=-1)  # (B, N, 3+C)
        x = self.net(x)
        return self.final(x)


# =============================================================================
# 3. 3D UNET COMPONENTS
# =============================================================================

class DoubleConv3D(nn.Module):
    """3D double convolution block"""
    def __init__(self, in_ch, out_ch, use_batchnorm=True):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_batchnorm)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm3d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=not use_batchnorm)
        )
        if use_batchnorm:
            layers.append(nn.BatchNorm3d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    """
    3D U-Net with skip connections
    Returns ONLY the final decoder output
    """
    def __init__(
        self,
        in_channels=1,
        base_channels=32,  # Smaller default for 3D due to memory
        depth=4,
        use_batchnorm=True
    ):
        super().__init__()
        self.depth = depth
        self.pool = nn.MaxPool3d(2)  # 3D pooling

        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = in_channels
        ch = base_channels

        for _ in range(depth):
            self.encoders.append(DoubleConv3D(in_ch, ch, use_batchnorm))
            in_ch = ch
            ch *= 2

        # Bottleneck
        self.bottleneck = DoubleConv3D(in_ch, ch, use_batchnorm)

        # Decoder
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        for _ in range(depth):
            ch //= 2
            self.upconvs.append(
                nn.ConvTranspose3d(ch * 2, ch, kernel_size=2, stride=2)  # 3D transpose conv
            )
            self.decoders.append(DoubleConv3D(ch * 2, ch, use_batchnorm))

    def forward(self, x):
        """
        Input:  (B, in_channels, D, H, W)
        Returns: (B, base_channels, D, H, W) - ONLY final decoder output
        """
        skips = []

        # Encoder path
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Decoder path with skip connections
        for i in range(self.depth):
            x = self.upconvs[i](x)
            skip = skips[-(i + 1)]
            
            # Handle size mismatches (copy and crop)
            if x.shape != skip.shape:
                diffD = skip.size()[2] - x.size()[2]
                diffH = skip.size()[3] - x.size()[3]
                diffW = skip.size()[4] - x.size()[4]
                x = F.pad(x, [
                    diffW // 2, diffW - diffW // 2,
                    diffH // 2, diffH - diffH // 2,
                    diffD // 2, diffD - diffD // 2
                ])
            
            x = torch.cat([x, skip], dim=1)
            x = self.decoders[i](x)

        return x  # Final output only


# =============================================================================
# 4. FULL 3D MODEL
# =============================================================================

class UNet3DWithINR(nn.Module):
    """
    3D U-Net + INR:
    - 3D U-Net extracts volumetric features
    - ONLY the final 3D U-Net output goes to INR
    - INR input = normalized 3D voxel coordinates + final features
    """
    def __init__(
        self,
        # UNet params
        in_channels=1,
        base_channels=32,  # Smaller for 3D
        unet_depth=4,
        use_batchnorm=True,

        # INR params
        inr_hidden_dim=64,  # Smaller for 3D
        inr_layers=3,
        inr_omega_0=30.0,
        
        # Feature projection
        use_feature_projection=True,
        projection_dim=8,  # Smaller for 3D

        # Task
        num_classes=3
    ):
        super().__init__()
        
        self.use_feature_projection = use_feature_projection

        # 3D U-Net backbone
        self.unet = UNet3D(
            in_channels=in_channels,
            base_channels=base_channels,
            depth=unet_depth,
            use_batchnorm=use_batchnorm
        )

        # Optional 1x1x1 conv to reduce feature dimension
        unet_out_channels = base_channels
        if use_feature_projection:
            self.feature_proj = nn.Conv3d(
                unet_out_channels, 
                projection_dim, 
                kernel_size=1
            )
            inr_feature_dim = projection_dim
        else:
            self.feature_proj = None
            inr_feature_dim = unet_out_channels

        # 3D INR head - receives 3D coordinates + final features
        self.inr = INRHead3D(
            feature_dim=inr_feature_dim,
            coord_dim=3,  # 3D coordinates!
            hidden_dim=inr_hidden_dim,
            num_layers=inr_layers,
            omega_0=inr_omega_0,
            num_classes=num_classes
        )
        
        # Coordinate cache for efficiency
        self.register_buffer('cached_coords', None)
        self.cached_size = None

    def get_coordinates(self, D, H, W, device):
        """
        Generate normalized 3D voxel coordinates in [-1, 1]
        Returns: (D*H*W, 3) tensor with (x, y, z) coordinates
        """
        if self.cached_coords is None or self.cached_size != (D, H, W):
            # Create normalized 3D grid coordinates
            zs, ys, xs = torch.meshgrid(
                torch.linspace(-1, 1, D, device=device),
                torch.linspace(-1, 1, H, device=device),
                torch.linspace(-1, 1, W, device=device),
                indexing="ij"
            )
            # Stack as (x, y, z) for consistency
            coords = torch.stack([xs, ys, zs], dim=-1).reshape(-1, 3)  # (D*H*W, 3)
            self.cached_coords = coords
            self.cached_size = (D, H, W)
        return self.cached_coords

    def forward(self, x):
        """
        x: (B, in_channels, D, H, W) - 3D volume
        Returns: (B, num_classes, D, H, W) - 3D segmentation
        """
        B, _, D, H, W = x.shape
        
        # Step 1: Extract features from 3D U-Net (final output only)
        features = self.unet(x)  # (B, C, D, H, W)
        
        # Step 2: Optional feature projection with 1x1x1 conv
        if self.use_feature_projection:
            features = self.feature_proj(features)  # (B, projection_dim, D, H, W)
        
        C = features.shape[1]
        
        # Step 3: Get normalized 3D voxel coordinates
        coords = self.get_coordinates(D, H, W, x.device)  # (D*H*W, 3)
        coords = coords.unsqueeze(0).expand(B, -1, -1)  # (B, D*H*W, 3)
        
        # Step 4: Reshape features for INR
        features = features.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)  # (B, D*H*W, C)
        
        # Step 5: Pass through INR (BATCH PROCESSING - NO LOOP!)
        logits = self.inr(coords, features)  # (B, D*H*W, num_classes)
        
        # Step 6: Reshape back to 3D spatial dimensions
        logits = logits.view(B, D, H, W, -1).permute(0, 4, 1, 2, 3)  # (B, num_classes, D, H, W)
        
        return logits


# =============================================================================
# 5. OPTIMIZED 3D CONFIGURATIONS
# =============================================================================

class UNet3DWithINRLightweight(UNet3DWithINR):
    """Lightweight 3D version - optimized for memory"""
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__(
            in_channels=in_channels,
            base_channels=16,           # Very small for 3D
            unet_depth=3,               # Shallower
            inr_hidden_dim=32,          # Smaller
            inr_layers=2,
            projection_dim=4,           # Minimal
            num_classes=num_classes,
            use_feature_projection=True
        )


class UNet3DWithINRBalanced(UNet3DWithINR):
    """Balanced 3D version"""
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__(
            in_channels=in_channels,
            base_channels=24,
            unet_depth=4,
            inr_hidden_dim=48,
            inr_layers=3,
            projection_dim=8,
            num_classes=num_classes,
            use_feature_projection=True
        )


class UNet3DWithINRPerformance(UNet3DWithINR):
    """Higher capacity 3D version"""
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__(
            in_channels=in_channels,
            base_channels=32,
            unet_depth=4,
            inr_hidden_dim=64,
            inr_layers=3,
            projection_dim=12,
            num_classes=num_classes,
            use_feature_projection=True
        )


# =============================================================================
# 6. MEMORY-EFFICIENT VERSION FOR LARGE 3D VOLUMES
# =============================================================================

class UNet3DWithINRMemoryEfficient(UNet3DWithINR):
    """
    For large 3D volumes, process INR in chunks to avoid OOM
    Critical for volumes like 256x256x256 or larger
    """
    def __init__(self, *args, chunk_size=8192, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_size = chunk_size

    def forward(self, x):
        B, _, D, H, W = x.shape
        
        # Extract features
        features = self.unet(x)
        if self.use_feature_projection:
            features = self.feature_proj(features)
        C = features.shape[1]
        
        # Get coordinates
        coords = self.get_coordinates(D, H, W, x.device)
        coords = coords.unsqueeze(0).expand(B, -1, -1)
        
        # Reshape features
        features = features.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)
        
        # Process in chunks (CRITICAL FOR 3D!)
        N = D * H * W
        if N > self.chunk_size:
            logits_list = []
            for i in range(0, N, self.chunk_size):
                end_idx = min(i + self.chunk_size, N)
                chunk_coords = coords[:, i:end_idx]
                chunk_features = features[:, i:end_idx]
                chunk_logits = self.inr(chunk_coords, chunk_features)
                logits_list.append(chunk_logits)
            logits = torch.cat(logits_list, dim=1)
        else:
            logits = self.inr(coords, features)
        
        logits = logits.view(B, D, H, W, -1).permute(0, 4, 1, 2, 3)
        return logits


# =============================================================================
# 7. UTILITY FUNCTIONS
# =============================================================================

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_3d_model():
    """Quick test of 3D model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = UNet3DWithINRBalanced(in_channels=1, num_classes=3).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass with small volume
    x = torch.randn(1, 1, 32, 64, 64).to(device)  # (B, C, D, H, W)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        out = model(x)
    
    print(f"Output shape: {out.shape}")
    print(f"Expected: (1, 3, 32, 64, 64)")
    
    assert out.shape == (1, 3, 32, 64, 64), "Output shape mismatch!"
    print("✓ Test passed!")


if __name__ == "__main__":
    test_3d_model()