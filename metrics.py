from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
#  Euclidean distance
# ─────────────────────────────────────────────────────────────────────────────
def euclidean_distance(pred_coords, gt_coords, visibility, image_size=224):
    pred = pred_coords * image_size
    gt   = gt_coords   * image_size
    vis  = visibility > 0
    diff = pred - gt
    dists = np.sqrt((diff**2).sum(-1))
    dists_masked = np.where(vis, dists, np.nan)
    mean_dist = float(np.nanmean(dists_masked))
    per_kp = [float(np.nanmean(dists_masked[:,k])) for k in range(dists.shape[1])]
    return {"mean_dist": mean_dist, "per_keypoint": per_kp}


# ─────────────────────────────────────────────────────────────────────────────
#  JSD
# ─────────────────────────────────────────────────────────────────────────────
def jsd_metric(pred_hm, gt_hm, eps=1e-6):
    N,K,H,W = pred_hm.shape
    pred_flat = pred_hm.reshape(N,K,-1)
    gt_flat   = gt_hm.reshape(N,K,-1)
    def softmax(x):
        e = np.exp(x - x.max(-1, keepdims=True))
        return e / (e.sum(-1, keepdims=True)+eps)
    p = softmax(pred_flat) + eps
    q = softmax(gt_flat)   + eps
    m = 0.5*(p+q)
    kl_pm = (p * np.log(p/m)).sum(-1)
    kl_qm = (q * np.log(q/m)).sum(-1)
    jsd = np.sqrt(0.5*(kl_pm+kl_qm)+eps)
    return float(jsd.mean())


# ─────────────────────────────────────────────────────────────────────────────
#  OKS
# ─────────────────────────────────────────────────────────────────────────────
def compute_oks(pred, gt, vis, scale, sigmas=None):
    K = len(gt)
    if sigmas is None:
        sigmas = np.full(K, 0.05, dtype=np.float64)
    mask = vis > 0
    if mask.sum() == 0:
        return 0.0
    d2 = ((pred - gt)**2).sum(-1)
    e = np.exp(-d2 / (2*(scale*sigmas)**2 + 1e-8))
    return float(e[mask].sum() / mask.sum())


def compute_oks_batch(pred, gt, vis, scales, sigmas=None):
    return np.array([compute_oks(pred[i], gt[i], vis[i], scales[i], sigmas) for i in range(len(pred))])


def compute_ap_ar(oks_values, thresholds=None):
    if thresholds is None:
        thresholds = [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]
    ap_vals, ar_vals = {}, {}
    for t in thresholds:
        correct = (oks_values >= t).astype(float)
        ap_vals[t] = float(correct.mean())
        ar_vals[t] = float(correct.mean())
    return {
        "AP": np.mean(list(ap_vals.values())),
        "AP_50": ap_vals.get(0.50, np.nan),
        "AP_75": ap_vals.get(0.75, np.nan),
        "AR": np.mean(list(ar_vals.values())),
        "AR_50": ar_vals.get(0.50, np.nan),
        "AR_75": ar_vals.get(0.75, np.nan),
        "per_threshold_ap": ap_vals,
        "per_threshold_ar": ar_vals,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Full evaluator
# ─────────────────────────────────────────────────────────────────────────────
class KeypointEvaluator:
    def __init__(self, image_size=224, sigmas=None):
        self.image_size = image_size
        self.sigmas = sigmas
        self._pred_hm, self._gt_hm = [], []
        self._pred_co, self._gt_co = [], []
        self._vis = []
        self._bbox = []            # store normalized bbox tuples (x1,y1,x2,y2)

    def reset(self):
        self._pred_hm.clear(); self._gt_hm.clear()
        self._pred_co.clear(); self._gt_co.clear()
        self._vis.clear(); self._bbox.clear()

    def update(self, pred_hm, pred_co, gt_hm, gt_co, visibility, bbox=None):
        self._pred_hm.append(pred_hm.cpu().numpy())
        self._gt_hm.append(gt_hm.cpu().numpy())
        self._pred_co.append(pred_co.detach().cpu().numpy())
        self._gt_co.append(gt_co.cpu().numpy())
        self._vis.append(visibility.cpu().numpy())
        if bbox is not None:
            # bbox can be a list/tensor of (B,4) normalized
            if isinstance(bbox, torch.Tensor):
                self._bbox.append(bbox.cpu().numpy())
            else:
                self._bbox.append(np.array(bbox))

    def compute(self):
        pred_hm = np.concatenate(self._pred_hm)
        gt_hm   = np.concatenate(self._gt_hm)
        pred_co = np.concatenate(self._pred_co)
        gt_co   = np.concatenate(self._gt_co)
        vis     = np.concatenate(self._vis)

        # Euclidean
        ed = euclidean_distance(pred_co, gt_co, vis, self.image_size)

        # JSD
        jsd = jsd_metric(pred_hm, gt_hm)

        # OKS scale: prefer bbox if available, else keypoint bounding box
        if len(self._bbox) > 0:
            bbox_all = np.concatenate(self._bbox)  # (N,4)
            scales = []
            for i in range(len(gt_co)):
                if bbox_all[i,0] != bbox_all[i,2]:   # valid box
                    w = (bbox_all[i,2] - bbox_all[i,0])  # normalized width
                    h = (bbox_all[i,3] - bbox_all[i,1])
                    area = w * h
                    scales.append(np.sqrt(max(area, 1e-8)))
                else:
                    # fallback to keypoints
                    valid = vis[i] > 0
                    if valid.sum() == 0:
                        scales.append(1.0)
                    else:
                        x = gt_co[i,valid,0]; y = gt_co[i,valid,1]
                        w = x.max()-x.min(); h = y.max()-y.min()
                        scales.append(np.sqrt(max(w*h, 1e-8)))
            scales = np.array(scales)
        else:
            scales = []
            for i in range(len(gt_co)):
                valid = vis[i] > 0
                if valid.sum() == 0:
                    scales.append(1.0)
                else:
                    x = gt_co[i,valid,0]; y = gt_co[i,valid,1]
                    w = x.max()-x.min(); h = y.max()-y.min()
                    scales.append(np.sqrt(max(w*h, 1e-8)))
            scales = np.array(scales)

        oks_vals = compute_oks_batch(pred_co, gt_co, vis, scales, self.sigmas)
        ap_ar = compute_ap_ar(oks_vals)

        return {
            "euclidean_mean_px": ed["mean_dist"],
            "euclidean_per_kp": ed["per_keypoint"],
            "jsd": jsd,
            "oks_mean": float(oks_vals.mean()),
            **ap_ar,
        }

    @staticmethod
    def print_results(results):
        print("\n" + "="*60)
        print(f"  AP      = {results['AP']:.3f}")
        print(f"  AP@.50  = {results['AP_50']:.3f}")
        print(f"  AP@.75  = {results['AP_75']:.3f}")
        print(f"  AR      = {results['AR']:.3f}")
        print(f"  AR@.50  = {results['AR_50']:.3f}")
        print(f"  AR@.75  = {results['AR_75']:.3f}")
        print(f"  OKS     = {results['oks_mean']:.3f}")
        print(f"  Euclid  = {results['euclidean_mean_px']:.2f} px")
        print(f"  JSD     = {results['jsd']:.4f}")
        print("="*60)