

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


#  Individual loss components
# 

class JSDHeatmapLoss(nn.Module):
  

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred:   torch.Tensor,   # (B, K, H, W)  raw logits
        target: torch.Tensor,   # (B, K, H, W)  raw logits / Gaussian heatmaps
    ) -> torch.Tensor:

        B, K, H, W = pred.shape
        pred_flat   = pred.view(B, K, -1)
        target_flat = target.view(B, K, -1)

        # Convert to probability distributions
        p = F.softmax(pred_flat,   dim=-1) + self.eps
        q = F.softmax(target_flat, dim=-1) + self.eps

        # Pointwise mean
        m = 0.5 * (p + q)

        # KL divergences  KL(P||M) and KL(Q||M)
        kl_pm = (p * (p / m).log()).sum(dim=-1)   # (B, K)
        kl_qm = (q * (q / m).log()).sum(dim=-1)   # (B, K)

        # JSD per (sample, keypoint)
        jsd = torch.sqrt(0.5 * (kl_pm + kl_qm) + self.eps)

        return jsd.mean()


class EuclideanCoordLoss(nn.Module):


    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred_coords: torch.Tensor,    # (B, K, 2)
        gt_coords:   torch.Tensor,    # (B, K, 2)
        visibility:  torch.Tensor,    # (B, K)  int  0/1/2
    ) -> torch.Tensor:

        diff  = pred_coords - gt_coords           # (B, K, 2)
        dist  = torch.sqrt((diff ** 2).sum(-1) + self.eps)   # (B, K)

        vis_mask = (visibility > 0).float()       # (B, K)
        n_visible = vis_mask.sum().clamp(min=1.0)

        return (dist * vis_mask).sum() / n_visible


class WingLoss(nn.Module):


    def __init__(self, w: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * torch.tensor(1.0 + w / epsilon).log()

    def forward(
        self,
        pred_coords: torch.Tensor,
        gt_coords:   torch.Tensor,
        visibility:  torch.Tensor,
    ) -> torch.Tensor:
        diff     = (pred_coords - gt_coords).abs().sum(-1)   # (B, K)
        inside   = diff < self.w
        loss     = torch.where(
            inside,
            self.w * (1.0 + diff / self.epsilon).log(),
            diff - self.C.to(diff.device),
        )
        vis_mask  = (visibility > 0).float()
        n_visible = vis_mask.sum().clamp(min=1.0)
        return (loss * vis_mask).sum() / n_visible


# ─────────────────────────────────────────────────────────────────────────────
#  Combined multi-task loss
# ─────────────────────────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):


    def __init__(
        self,
        alpha: float = config.MULTITASK_ALPHA,
        use_wing_loss: bool = False,
    ):
        super().__init__()
        self.alpha       = alpha
        self.jsd_loss    = JSDHeatmapLoss()
        self.coord_loss  = WingLoss() if use_wing_loss else EuclideanCoordLoss()

    def forward(
        self,
        pred_heatmap:  torch.Tensor,   # (B, K, H, W)
        pred_coords:   torch.Tensor,   # (B, K, 2)
        gt_heatmap:    torch.Tensor,   # (B, K, H, W)
        gt_coords:     torch.Tensor,   # (B, K, 2)
        visibility:    torch.Tensor,   # (B, K)
    ):
        # Returns
        # -------
        # total_loss   : scalar Tensor
        # coord_loss   : scalar Tensor  (for logging)
        # heatmap_loss : scalar Tensor  (for logging)
        heatmap_loss = self.jsd_loss(pred_heatmap, gt_heatmap)
        coord_loss   = self.coord_loss(pred_coords, gt_coords, visibility)

        total = self.alpha * heatmap_loss + (1.0 - self.alpha) * coord_loss

        return total, coord_loss.detach(), heatmap_loss.detach()


# ─────────────────────────────────────────────────────────────────────────────
#  Sanity checker
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, K, H, W = 4, 16, 56, 56

    pred_hm  = torch.randn(B, K, H, W)
    gt_hm    = torch.randn(B, K, H, W).abs()  
    pred_co  = torch.rand(B, K, 2)
    gt_co    = torch.rand(B, K, 2)
    vis      = torch.ones(B, K, dtype=torch.long)

    criterion = MultiTaskLoss()
    total, c_loss, h_loss = criterion(pred_hm, pred_co, gt_hm, gt_co, vis)
    print(f"total={total:.4f}  coord={c_loss:.4f}  heatmap={h_loss:.4f}")