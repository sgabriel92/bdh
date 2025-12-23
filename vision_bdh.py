# Copyright 2025 Pathway Technology, Inc.
# Vision-BDH: BDH architecture adapted for images (like ViT)
# Key changes: patch embedding, 2D positional encoding, bidirectional attention

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class VisionBDHConfig:
    """Configuration for Vision-BDH model."""
    img_size: int = 32           # Input image size (32 for CIFAR, 224 for ImageNet)
    patch_size: int = 4          # Patch size (4 for CIFAR, 16 for ImageNet)
    in_channels: int = 3         # RGB
    n_classes: int = 10          # Number of output classes
    
    n_layer: int = 6             # L: number of BDH layers
    n_embd: int = 256            # D: embedding dimension
    n_head: int = 4              # H: number of attention heads
    n_neurons: int = 8192        # N: sparse neuron count
    dropout: float = 0.1
    
    @property
    def n_patches(self):
        return (self.img_size // self.patch_size) ** 2
    
    @property
    def grid_size(self):
        return self.img_size // self.patch_size


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    Generate 2D sinusoidal positional embeddings.
    
    Args:
        embed_dim: embedding dimension
        grid_size: int, grid height and width
    Returns:
        pos_embed: [grid_size*grid_size, embed_dim]
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(grid_h, grid_w, indexing='ij'), dim=-1)  # H, W, 2
    grid = grid.reshape(-1, 2)  # HW, 2
    
    # Split embedding dimension for x and y
    half_dim = embed_dim // 2
    emb_h = get_1d_sincos_pos_embed(half_dim, grid[:, 0])
    emb_w = get_1d_sincos_pos_embed(half_dim, grid[:, 1])
    
    pos_embed = torch.cat([emb_h, emb_w], dim=-1)  # HW, D
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, positions):
    """
    Generate 1D sinusoidal positional embeddings.
    
    Args:
        embed_dim: embedding dimension
        positions: [N] positions
    Returns:
        pos_embed: [N, embed_dim]
    """
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim // 2)))
    
    positions = positions.unsqueeze(-1)  # N, 1
    out = positions * omega.unsqueeze(0)  # N, D/2
    
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    
    return torch.cat([emb_sin, emb_cos], dim=-1)  # N, D


class BidirectionalAttention(nn.Module):
    """
    Bidirectional linear attention for vision (no causal mask).
    All patches can attend to all other patches.
    """
    
    def __init__(self, config: VisionBDHConfig):
        super().__init__()
        self.config = config
        N_per_head = config.n_neurons // config.n_head
        
        # 2D-aware frequency encoding
        self.freqs = nn.Buffer(
            self._get_2d_freqs(N_per_head, config.grid_size)
        )
    
    def _get_2d_freqs(self, dim, grid_size):
        """Generate 2D frequency components for RoPE."""
        # Split frequencies for row and column
        half_dim = dim // 2
        theta = 10000.0
        
        freqs_h = 1.0 / (theta ** (torch.arange(0, half_dim, 2).float() / half_dim))
        freqs_w = 1.0 / (theta ** (torch.arange(0, half_dim, 2).float() / half_dim))
        
        # Create position grids
        h_pos = torch.arange(grid_size).float()
        w_pos = torch.arange(grid_size).float()
        
        # Compute 2D phases: grid_size × grid_size × half_dim
        phases_h = torch.einsum('h,d->hd', h_pos, freqs_h)  # grid_size, half_dim//2
        phases_w = torch.einsum('w,d->wd', w_pos, freqs_w)  # grid_size, half_dim//2
        
        # Combine into full position encoding
        # For each (h, w) position, concatenate h-phases and w-phases
        phases = torch.zeros(grid_size, grid_size, dim)
        for h in range(grid_size):
            for w in range(grid_size):
                phases[h, w, :half_dim//2] = phases_h[h]
                phases[h, w, half_dim//2:half_dim] = phases_w[w]
                phases[h, w, half_dim:half_dim + half_dim//2] = phases_h[h]
                phases[h, w, half_dim + half_dim//2:] = phases_w[w]
        
        return phases.reshape(grid_size * grid_size, dim).unsqueeze(0).unsqueeze(0)
    
    @staticmethod
    def rope(phases, v):
        """Apply rotary position embeddings."""
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)
    
    def forward(self, Q, K, V):
        """
        Bidirectional attention (no causal mask).
        
        Args:
            Q, K: B, H, T, N//H (sparse activations)
            V: B, 1, T, D (dense representation)
        Returns:
            B, H, T, D
        """
        B, H, T, N_per_H = Q.size()
        n_patches = self.config.grid_size ** 2
        
        # Handle CLS token: first position has no 2D structure
        has_cls = (T == n_patches + 1)
        
        if has_cls:
            # Split CLS and patches
            Q_cls, Q_patches = Q[:, :, :1, :], Q[:, :, 1:, :]
            K_cls, K_patches = K[:, :, :1, :], K[:, :, 1:, :]
            
            # Apply 2D RoPE only to patches
            freqs = self.freqs[:, :, :n_patches, :N_per_H].to(Q.device)
            Qr_patches = self.rope(freqs, Q_patches)
            Kr_patches = self.rope(freqs, K_patches)
            
            # CLS token: no RoPE (position-agnostic)
            Qr = torch.cat([Q_cls, Qr_patches], dim=2)
            Kr = torch.cat([K_cls, Kr_patches], dim=2)
        else:
            # No CLS token, apply RoPE to all
            freqs = self.freqs[:, :, :T, :N_per_H].to(Q.device)
            Qr = self.rope(freqs, Q)
            Kr = self.rope(freqs, K)
        
        # Bidirectional attention: NO causal mask!
        scores = Qr @ Kr.mT  # B, H, T, T
        
        # Optional: normalize for stability
        scores = scores / math.sqrt(N_per_H)
        
        return scores @ V  # B, H, T, D


class VisionBDH(nn.Module):
    """
    Vision-BDH: BDH architecture for image classification.
    
    Key differences from text BDH:
    1. Patch embedding instead of token embedding
    2. 2D positional encoding
    3. Bidirectional attention (all patches see all patches)
    4. Classification head using [CLS] token or global average pooling
    """
    
    def __init__(self, config: VisionBDHConfig):
        super().__init__()
        self.config = config
        
        D = config.n_embd
        H = config.n_head
        N = config.n_neurons
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            config.in_channels, D,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
        
        # 2D positional embedding (learned, can also use sinusoidal)
        n_positions = config.n_patches + 1  # +1 for CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, n_positions, D))
        
        # Initialize with sinusoidal (better than random)
        self._init_pos_embed()
        
        # Layer norm
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        
        # Dropout
        self.drop = nn.Dropout(config.dropout)
        
        # BDH projections (shared across layers)
        self.decoder_x = nn.Parameter(torch.zeros((H, D, N // H)).normal_(std=0.02))
        self.decoder_y = nn.Parameter(torch.zeros((H, D, N // H)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((N, D)).normal_(std=0.02))
        
        # Bidirectional attention
        self.attn = BidirectionalAttention(config)
        
        # Classification head
        self.head = nn.Linear(D, config.n_classes)
        
        # Initialize
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_pos_embed(self):
        """Initialize positional embeddings with 2D sinusoidal."""
        # Get sinusoidal embeddings for patches
        pos_embed = get_2d_sincos_pos_embed(
            self.config.n_embd, 
            self.config.grid_size
        )
        # Add zeros for CLS token position
        cls_pos = torch.zeros(1, self.config.n_embd)
        full_pos = torch.cat([cls_pos, pos_embed], dim=0)
        self.pos_embed.data.copy_(full_pos.unsqueeze(0))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def patchify(self, img):
        """Convert image to patch sequence.
        
        Args:
            img: B, C, H, W
        Returns:
            patches: B, n_patches, D
        """
        # B, D, grid_h, grid_w
        x = self.patch_embed(img)
        # B, D, n_patches -> B, n_patches, D
        x = x.flatten(2).transpose(1, 2)
        return x
    
    def forward(self, img, targets=None):
        """
        Args:
            img: B, C, H, W input images
            targets: B class labels (optional, for loss)
        Returns:
            logits: B, n_classes
            loss: scalar if targets provided
        """
        config = self.config
        B = img.size(0)
        D = config.n_embd
        H = config.n_head
        N = config.n_neurons
        L = config.n_layer
        
        # Patchify: B, n_patches, D
        x = self.patchify(img)
        
        # Add CLS token: B, n_patches+1, D
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        T = x.size(1)
        
        # Add positional embedding
        x = x + self.pos_embed[:, :T]
        x = self.drop(x)
        
        # Add head dimension for BDH: B, 1, T, D
        v_ast = self.ln(x).unsqueeze(1)
        
        # BDH layers
        for _ in range(L):
            # Project to sparse: B, H, T, N//H
            sparse_x = F.relu(v_ast @ self.decoder_x)
            
            # Bidirectional attention
            a_ast = self.attn(Q=sparse_x, K=sparse_x, V=v_ast)  # B, H, T, D
            
            # Gated sparse output
            sparse_y = F.relu(self.ln(a_ast) @ self.decoder_y) * sparse_x  # B, H, T, N//H
            
            # Reshape and project back
            y = sparse_y.transpose(1, 2).reshape(B, 1, T, N)  # B, 1, T, N
            y = self.drop(y)
            
            # Residual connection
            v_ast = v_ast + self.ln(y @ self.encoder)
        
        # Final layer norm
        v_ast = self.ln(v_ast)
        
        # Classification: use CLS token
        cls_output = v_ast[:, 0, 0, :]  # B, D
        logits = self.head(cls_output)  # B, n_classes
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def get_sparse_activations(self, img):
        """
        Extract sparse activations for visualization.
        Returns activations from the first layer.
        """
        config = self.config
        B = img.size(0)
        
        # Patchify
        x = self.patchify(img)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        T = x.size(1)
        
        x = x + self.pos_embed[:, :T]
        v_ast = self.ln(x).unsqueeze(1)
        
        # Get first layer sparse activations
        sparse_x = F.relu(v_ast @ self.decoder_x)  # B, H, T, N//H
        
        # Flatten heads: B, T, N
        sparse_flat = sparse_x.transpose(1, 2).reshape(B, T, -1)
        
        return sparse_flat
    
    @torch.no_grad()
    def predict(self, img):
        """Simple prediction method."""
        self.eval()
        logits, _ = self(img)
        return logits.argmax(dim=-1)


# Convenience function to create configs for common datasets
def cifar10_config():
    """Config for CIFAR-10 (32x32 images)."""
    return VisionBDHConfig(
        img_size=32,
        patch_size=4,      # 8x8 = 64 patches
        in_channels=3,
        n_classes=10,
        n_layer=6,
        n_embd=256,
        n_head=4,
        n_neurons=8192,
        dropout=0.1,
    )


def cifar100_config():
    """Config for CIFAR-100 (32x32 images)."""
    return VisionBDHConfig(
        img_size=32,
        patch_size=4,
        in_channels=3,
        n_classes=100,
        n_layer=8,
        n_embd=384,
        n_head=6,
        n_neurons=16384,
        dropout=0.1,
    )


def imagenet_config():
    """Config for ImageNet (224x224 images)."""
    return VisionBDHConfig(
        img_size=224,
        patch_size=16,     # 14x14 = 196 patches
        in_channels=3,
        n_classes=1000,
        n_layer=12,
        n_embd=768,
        n_head=12,
        n_neurons=65536,
        dropout=0.1,
    )


if __name__ == "__main__":
    # Quick test
    print("Testing Vision-BDH...")
    
    config = cifar10_config()
    model = VisionBDH(config)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Patches: {config.n_patches} ({config.grid_size}×{config.grid_size})")
    print(f"Neurons: {config.n_neurons:,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    logits, _ = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {logits.shape}")
    
    # Test with loss
    targets = torch.randint(0, 10, (2,))
    logits, loss = model(x, targets)
    print(f"Loss: {loss.item():.4f}")
    
    # Test sparse activations
    sparse = model.get_sparse_activations(x)
    print(f"Sparse activations: {sparse.shape}")
    sparsity = (sparse == 0).float().mean()
    print(f"Sparsity: {sparsity:.1%}")
    
    print("\n✓ Vision-BDH working!")

