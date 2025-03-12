import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from attention import FlashAttention, SlidingWindowAttention

class PatchEmbedding(nn.Module):
    """
    Image to Patch Embedding with overlapping patches
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, stride=None):
        super().__init__()
        stride = stride or patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // stride
        self.num_patches = self.grid_size ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2 if stride < patch_size else 0
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, grid_size, grid_size]
        x = rearrange(x, 'b c h w -> b (h w) c')  # [B, num_patches, embed_dim]
        x = self.norm(x)
        return x

class ViTBlock(nn.Module):
    """
    Vision Transformer Block with Flash Attention
    """
    def __init__(self, dim, heads=8, dim_head=64, mlp_ratio=4, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = FlashAttention(dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class SimpleViT(nn.Module):
    """
    Simple Vision Transformer with Flash Attention
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        heads=12,
        dim_head=64,
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Add cls token and position embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, heads, dim_head, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add cls token and position embedding
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        x = x[:, 0]  # Take cls token
        x = self.head(x)
        return x

class AttentionConvBlock(nn.Module):
    """
    CNN block with attention for feature refinement
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Attention module
        self.attention = FlashAttention(out_channels, heads=8, dim_head=out_channels//8)
        
    def forward(self, x):
        # Convolution path
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        # Attention path
        B, C, H, W = x.shape
        attention_in = rearrange(x, 'b c h w -> b (h w) c')
        attention_out = self.attention(attention_in)
        attention_out = rearrange(attention_out, 'b (h w) c -> b c h w', h=H, w=W)
        
        # Residual connection
        x = x + attention_out
        x = F.relu(x)
        return x

class HybridCNNAttention(nn.Module):
    """
    Hybrid CNN model with attention for feature extraction
    """
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Attention-enhanced conv blocks
        self.layer1 = AttentionConvBlock(64, 128, stride=2)
        self.layer2 = AttentionConvBlock(128, 256, stride=2)
        self.layer3 = AttentionConvBlock(256, 512, stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Initial convolution
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        
        # Attention-enhanced conv blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def test_vit():
    """
    Test Vision Transformer with Flash Attention
    """
    print("\n=== Testing Vision Transformer ===")
    
    # Create sample input
    batch_size = 4
    img_size = 224
    in_channels = 3
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    
    # Initialize ViT
    model = SimpleViT(
        img_size=img_size,
        patch_size=16,
        in_channels=in_channels,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        heads=12
    )
    
    # Forward pass
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"ViT check passed: {out.shape == (batch_size, 1000)}")

def test_hybrid_cnn():
    """
    Test Hybrid CNN with Attention
    """
    print("\n=== Testing Hybrid CNN-Attention ===")
    
    # Create sample input
    batch_size = 4
    img_size = 224
    in_channels = 3
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    
    # Initialize model
    model = HybridCNNAttention(in_channels=in_channels, num_classes=1000)
    
    # Forward pass
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Hybrid CNN check passed: {out.shape == (batch_size, 1000)}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run tests
    test_vit()
    test_hybrid_cnn()