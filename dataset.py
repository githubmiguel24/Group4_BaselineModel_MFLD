import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config
from utils import generate_gaussian_heatmap


def build_val_transforms(input_size: int = config.INPUT_SIZE):
    return A.Compose(
        [
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xyas", remove_invisible=False, angle_in_degrees=True),
    )


def parse_cvat_xml(xml_path: str, label_to_id: dict) -> Dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = {}
    for image_elem in root.findall('image'):
        raw_filename = image_elem.get('name')
        filename = os.path.basename(raw_filename.replace('\\', '/'))
        width  = int(image_elem.get('width'))
        height = int(image_elem.get('height'))

        coords     = np.zeros((config.NUM_KEYPOINTS, 2), dtype=np.float32)
        visibility = np.zeros(config.NUM_KEYPOINTS, dtype=np.int32)

        for points_elem in image_elem.findall('points'):
            label = points_elem.get('label')
            pts   = points_elem.get('points')
            occl  = int(points_elem.get('occluded', '0'))
            if label in label_to_id:
                kp_id = label_to_id[label]
                first = pts.split(';')[0]
                x, y = map(float, first.split(','))
                coords[kp_id] = [x, y]
                visibility[kp_id] = 1 if occl == 1 else 2

        # --- Fish_Box bounding box ---
        bbox = None
        for box_elem in image_elem.findall('box'):
            if box_elem.get('label') == 'Fish_Box':
                xtl = float(box_elem.get('xtl'))
                ytl = float(box_elem.get('ytl'))
                xbr = float(box_elem.get('xbr'))
                ybr = float(box_elem.get('ybr'))
                bbox = (xtl, ytl, xbr, ybr)
                break

        data[filename] = {
            "width":      width,
            "height":     height,
            "keypoints":  coords,
            "visibility": visibility,
            "bbox":       bbox,
        }
    return data


class FishKeypointDataset(Dataset):
    def __init__(self, image_dir: str, cvat_data: Dict, file_list: List[str],
                 transform=None, input_size=config.INPUT_SIZE,
                 heatmap_size=config.HEATMAP_SIZE, sigma=config.GAUSSIAN_SIGMA):
        self.image_dir = Path(image_dir)
        self.cvat_data = cvat_data
        self.file_list = file_list
        self.transform = transform
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        filename = self.file_list[idx]
        img_path = self.image_dir / filename

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann = self.cvat_data[filename]
        keypoints  = ann["keypoints"].copy()
        visibility = ann["visibility"].copy()
        bbox_raw   = ann.get("bbox")          # (x1,y1,x2,y2) in pixels, or None

        if self.transform is not None:
            kp_list = [(float(keypoints[k, 0]), float(keypoints[k, 1]), 0, 1)
                       for k in range(config.NUM_KEYPOINTS)]
            result = self.transform(image=image, keypoints=kp_list)
            image = result["image"]
            kp_out = result["keypoints"]
            new_kps = np.zeros((config.NUM_KEYPOINTS, 2), dtype=np.float32)
            new_vis = visibility.copy()
            for k in range(config.NUM_KEYPOINTS):
                if k < len(kp_out):
                    new_kps[k] = [kp_out[k][0], kp_out[k][1]]
                else:
                    new_vis[k] = 0
            keypoints, visibility = new_kps, new_vis
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        else:
            # manual resize + normalize (fallback)
            image = cv2.resize(image, (self.input_size, self.input_size))
            sx = self.input_size / ann["width"]
            sy = self.input_size / ann["height"]
            keypoints[:, 0] *= sx
            keypoints[:, 1] *= sy
            if bbox_raw is not None:
                x1, y1, x2, y2 = bbox_raw
                bbox_raw = (x1*sx, y1*sy, x2*sx, y2*sy)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            image = (image.astype(np.float32) / 255.0 - mean) / std
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        heatmaps = generate_gaussian_heatmap(
            keypoints, visibility,
            heatmap_size=self.heatmap_size,
            img_size=self.input_size,
            sigma=self.sigma,
        )

        norm_coords = keypoints.copy()
        norm_coords[:, 0] /= self.input_size
        norm_coords[:, 1] /= self.input_size

        # ---- ALWAYS return a (4,) tensor for the bounding box ----
        if bbox_raw is not None:
            bbox_norm = torch.tensor([
                bbox_raw[0] / self.input_size,
                bbox_raw[1] / self.input_size,
                bbox_raw[2] / self.input_size,
                bbox_raw[3] / self.input_size,
            ], dtype=torch.float32)
        else:
            # Missing box → zeros (will be ignored by evaluator)
            bbox_norm = torch.zeros(4, dtype=torch.float32)
        # ---------------------------------------------------------

        return (
            image,
            torch.from_numpy(heatmaps).float(),
            torch.from_numpy(norm_coords).float(),
            torch.from_numpy(visibility).long(),
            str(img_path),
            bbox_norm,          # now always a tensor
        )
    
def build_dataloaders(image_dir=config.IMAGE_DIR, annot_file=config.ANNOT_FILE,
                      batch_size=config.BATCH_SIZE, num_workers=12,
                      seed=config.RANDOM_SEED):
    label_to_id = {name: i for i, name in enumerate(config.KEYPOINT_NAMES)}
    cvat_data = parse_cvat_xml(annot_file, label_to_id)

    def read_split(txt_path):
        with open(txt_path) as f:
            return [line.strip() for line in f if line.strip()]

    split_dir = os.path.dirname(annot_file)
    train_files = read_split(os.path.join(split_dir, "train_files.txt"))
    val_files   = read_split(os.path.join(split_dir, "val_files.txt"))
    test_files  = read_split(os.path.join(split_dir, "test_files.txt"))

    transform = build_val_transforms(config.INPUT_SIZE)

    train_ds = FishKeypointDataset(image_dir, cvat_data, train_files, transform)
    val_ds   = FishKeypointDataset(image_dir, cvat_data, val_files,   transform)
    test_ds  = FishKeypointDataset(image_dir, cvat_data, test_files,  transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True)
    return train_loader, val_loader, test_loader