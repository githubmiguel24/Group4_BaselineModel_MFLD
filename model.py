import torch
import torch.nn as nn
import torch.nn.functional as F

import config


#patch Embedding - for a single large-stride convolution (kernel size 4, stride 4)

class PatchEmbedding(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim:   int = config.EMBED_DIM,
        patch_size:  int = config.PATCH_SIZE,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=False,
        )
        self.norm = nn.BatchNorm2d(embed_dim)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.proj(x)))


#convblocks base
class ConvBlock(nn.Module):

    def __init__(
        self,
        channels:     int   = config.EMBED_DIM,
        kernel_size:  int   = config.KERNEL_SIZE,
        dropout_rate: float = config.DROPOUT_RATE,
    ):
        super().__init__()
        pad = kernel_size // 2

        # Depthwise convolution  (spatial mixing)
        self.dw_conv = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size, padding=pad,
            groups=channels, bias=False,
        )
        self.dw_act  = nn.GELU()
        self.dw_norm = nn.BatchNorm2d(channels)

        # Spatial (2-D) dropout  – drops entire feature maps
        self.spatial_dropout = nn.Dropout2d(p=dropout_rate)

        # Pointwise convolution  (channel mixing)
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.pw_act  = nn.GELU()
        self.pw_norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Depthwise branch with residual ────────────────────────────────
        residual = x
        out = self.dw_conv(x)
        out = self.dw_act(out)
        out = self.dw_norm(out)
        out = out + residual           # element-wise add (residual connection)

        # ── Spatial dropout ───────────────────────────────────────────────
        out = self.spatial_dropout(out)

        # ── Pointwise branch ──────────────────────────────────────────────
        out = self.pw_conv(out)
        out = self.pw_act(out)
        out = self.pw_norm(out)

        return out


#  Soft-Argmax - for differentiable coordinate extraction from heatmaps
class SoftArgmax2D(nn.Module):
    """
    Converts a batch of heatmaps into (x, y) coordinate predictions.
    Coordinates are normalised to [0, 1].
    """

    def __init__(self, heatmap_size: int = config.HEATMAP_SIZE, temperature: float = 0.3):
        super().__init__()
        # Pre-compute normalised 1-D grids and register as non-trainable buffers
        coords = torch.linspace(0.0, 1.0, heatmap_size)
        self.register_buffer("x_coords", coords)   # W dimension
        self.register_buffer("y_coords", coords)   # H dimension
        self.temperature = temperature # ADD THIS: Temperature scaling factor

    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        B, K, H, W = heatmap.shape
        # Sharpen: divide by temperature (<1 makes logits larger → sharper peaks)
        prob = F.softmax(heatmap.view(B, K, -1) / self.temperature, dim=-1).view(B, K, H, W)
        x = (prob.sum(dim=2) * self.x_coords).sum(dim=-1)
        y = (prob.sum(dim=3) * self.y_coords).sum(dim=-1)
        return torch.stack([x, y], dim=2)

#  MFLD-net
class MFLDNet(nn.Module):

    def __init__(
        self,
        num_keypoints: int   = config.NUM_KEYPOINTS,
        embed_dim:     int   = config.EMBED_DIM,
        num_blocks:    int   = config.NUM_CONV_BLOCKS,
        kernel_size:   int   = config.KERNEL_SIZE,
        dropout_rate:  float = config.DROPOUT_RATE,
        heatmap_size:  int   = config.HEATMAP_SIZE,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.heatmap_size  = heatmap_size

        #Patch embedding  (stride-4 conv) 
        self.patch_embedding = PatchEmbedding(
            in_channels=3,
            embed_dim=embed_dim,
            patch_size=config.PATCH_SIZE,
        )

        #8 Isometric ConvBlocks 
        self.conv_blocks = nn.Sequential(
            *[ConvBlock(embed_dim, kernel_size, dropout_rate)
              for _ in range(num_blocks)]
        )

        #Heatmap head ─
        self.heatmap_conv = nn.Conv2d(embed_dim, num_keypoints, kernel_size=1)

        # ── Soft-Argmax (coordinate head) ─────────────────────────────────
        self.soft_argmax = SoftArgmax2D(heatmap_size, temperature=0.3)

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        # (B,3,224,224) → (B,256,56,56)
        features = self.patch_embedding(x)

        # (B,256,56,56) → (B,256,56,56)   isometric
        features = self.conv_blocks(features)

        # (B,256,56,56) → (B,K,56,56)
        heatmap = self.heatmap_conv(features)

        # (B,K,56,56) → (B,K,2)
        coords = self.soft_argmax(heatmap)

        return heatmap, coords

    # ── Convenience: count parameters ─────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_summary(self):
        total = self.count_parameters()
        print(f"MFLD-net  –  trainable parameters: {total / 1e6:.2f} M")
        for name, module in self.named_children():
            n = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"  {name:<25s}  {n / 1e6:.4f} M")


# ─────────────────────────────────────────────────────────────────────────────
#  Quick sanity-check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = MFLDNet()
    model.parameter_summary()

    dummy = torch.randn(2, 3, 224, 224)
    hm, co = model(dummy)
    print(f"\nForward pass OK")
    print(f"  heatmap : {hm.shape}")   # (2, 16, 56, 56)
    print(f"  coords  : {co.shape}")   # (2, 16, 2)