import os, torch, numpy as np, cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

import config
from model import MFLDNet
from metrics import KeypointEvaluator       # <-- uses the paper's metrics (Euclid, JSD, OKS, AP, AR)

# ── Settings ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "overfit_model.pth"
TEST_LIST  = "data/augmented/test_files.txt"
ORIG_IMAGE_DIR = "data/images"
ORIG_XML = "data/annotations.xml"

# ── Dataset (unchanged – returns normalized coords and visibility) ─
class TestDataset(Dataset):
    def __init__(self, image_dir, xml_path, file_list, input_size=224):
        self.image_dir = Path(image_dir)
        self.input_size = input_size
        tree = ET.parse(xml_path)
        root = tree.getroot()
        self.data = {}
        for img_elem in root.findall('image'):
            fname = os.path.basename(img_elem.get('name').replace('\\', '/'))
            w, h = int(img_elem.get('width')), int(img_elem.get('height'))
            coords = np.zeros((config.NUM_KEYPOINTS,2), dtype=np.float32)
            vis = np.zeros(config.NUM_KEYPOINTS, dtype=np.int32)
            for pt in img_elem.findall('points'):
                label = pt.get('label')
                pts_str = pt.get('points','')
                occluded = int(pt.get('occluded','0'))
                if label in label_to_id:
                    idx = label_to_id[label]
                    first = pts_str.split(';')[0]
                    x,y = map(float, first.split(','))
                    coords[idx] = [x,y]
                    vis[idx] = 1 if occluded==1 else 2
            self.data[fname] = (coords, vis)
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        img = cv2.imread(str(self.image_dir / fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        # resize to model input size
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_resized.astype(np.float32)/255.0 - mean) / std
        img_tensor = torch.from_numpy(img_norm.transpose(2,0,1)).float()

        coords, vis = self.data[fname]
        coords = coords.copy()
        coords[:,0] *= (self.input_size / orig_w)
        coords[:,1] *= (self.input_size / orig_h)
        norm_coords = coords / self.input_size   # [0,1]
        return img_tensor, torch.from_numpy(norm_coords).float(), torch.from_numpy(vis).long(), fname, orig_w, orig_h

# ── Prepare data ─────────────────────────────────────────
label_to_id = {name:i for i,name in enumerate(config.KEYPOINT_NAMES)}

with open(TEST_LIST) as f:
    test_raw = [line.strip() for line in f if line.strip()]
test_orig = [fn.replace('test_', '', 1) for fn in test_raw]
print(f"Test set: {len(test_orig)} images")

test_ds = TestDataset(ORIG_IMAGE_DIR, ORIG_XML, test_orig)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

# ── Load model ───────────────────────────────────────────
model = MFLDNet(
    num_keypoints=config.NUM_KEYPOINTS,
    embed_dim=512,
    num_blocks=12,
    kernel_size=config.KERNEL_SIZE,
    dropout_rate=0.0,
    heatmap_size=config.HEATMAP_SIZE,
).to(DEVICE)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded.\n")

# ── Evaluate with paper's metrics ─────────────────────────
evaluator = KeypointEvaluator(image_size=config.INPUT_SIZE, sigmas=config.OKS_SIGMAS)
example_saved = 0
output_dir = Path("test_output")
output_dir.mkdir(exist_ok=True)

with torch.no_grad():
    for batch_idx, (img_tensor, gt_coords_norm, vis, fname, orig_w, orig_h) in enumerate(test_loader):
        img_tensor = img_tensor.to(DEVICE)
        pred_hm, pred_co_norm = model(img_tensor)

        # Accumulate for metric computation
        # pred_hm is raw logits (B,K,56,56), gt_heatmaps not needed? The evaluator expects both.
        from utils import generate_gaussian_heatmap
        kp_px = (gt_coords_norm * config.INPUT_SIZE).cpu().numpy()  # (B,K,2)
        vis_np = vis.cpu().numpy()
        batch_size = kp_px.shape[0]
        gt_hm_list = []
        for i in range(batch_size):
            hm = generate_gaussian_heatmap(kp_px[i], vis_np[i],
                                           config.HEATMAP_SIZE, config.INPUT_SIZE,
                                           config.GAUSSIAN_SIGMA)
            gt_hm_list.append(torch.from_numpy(hm).unsqueeze(0))
        gt_hm = torch.cat(gt_hm_list, dim=0).float().to(DEVICE)

        evaluator.update(pred_hm, pred_co_norm, gt_hm, gt_coords_norm, vis)

        # Save a few annotated examples (first 3 images)
        if example_saved < 20:
            orig_img = cv2.imread(str(Path(ORIG_IMAGE_DIR) / fname[0]))
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            orig_h_i, orig_w_i = orig_img.shape[:2]

            pred_px = pred_co_norm[0].cpu().numpy().copy()
            pred_px[:,0] *= orig_w_i
            pred_px[:,1] *= orig_h_i

            gt_px = gt_coords_norm[0].cpu().numpy().copy()
            gt_px[:,0] *= orig_w_i
            gt_px[:,1] *= orig_h_i
            visibility = vis[0]

            plt.figure(figsize=(10, 8))
            plt.imshow(orig_img)
            for k in range(config.NUM_KEYPOINTS):
                if visibility[k] > 0:
                    plt.plot(gt_px[k,0], gt_px[k,1], 'go', markersize=8, label='GT' if k==0 else "")
                    plt.plot(pred_px[k,0], pred_px[k,1], 'r+', markersize=12, label='Pred' if k==0 else "")
            plt.legend()
            plt.title(f"Test image: {fname[0]}")
            save_path = output_dir / f"test_example_{example_saved+1}.png"
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"Saved example to {save_path}")
            example_saved += 1

# ── Compute and print standard metrics ───────────────────
results = evaluator.compute()
print("\n" + "="*60)
print("PAPER-STANDARD EVALUATION METRICS ON TEST SET")
print("="*60)
print(f"  AP      = {results['AP']:.3f}")
print(f"  AP@.50  = {results['AP_50']:.3f}")
print(f"  AP@.75  = {results['AP_75']:.3f}")
print(f"  AR      = {results['AR']:.3f}")
print(f"  AR@.50  = {results['AR_50']:.3f}")
print(f"  AR@.75  = {results['AR_75']:.3f}")
print(f"  OKS mean= {results['oks_mean']:.3f}")
print(f"  Euclid  = {results['euclidean_mean_px']:.2f} px")
print(f"  JSD     = {results['jsd']:.4f}")
print("="*60)
print(f"\nAnnotated test images saved in '{output_dir}' folder.")