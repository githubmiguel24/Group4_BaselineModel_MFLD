#!/usr/bin/env python3
"""
augment_and_split.py
--------------------
1. Parse & clean original CVAT XML (13 keypoints + Fish_Box).
2. Split into train / val / test sets.
3. Offline augmentation (train only) with bounding box scaling.
4. Write augmented images, XML, and split lists.
"""

import os, copy, random, shutil
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_XML      = "./data/annotations.xml"
IMAGE_DIR      = "./data/images"
OUTPUT_DIR     = "./data/augmented"
AUG_PER_IMAGE  = 10
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
TEST_RATIO     = 0.15
RANDOM_SEED    = 42

# Keypoint names (13 points – Ventral Base & Ventral Tip removed)
KEYPOINT_NAMES = [
    "Snout tip",
    "Eye center",
    "Dorsal fin base — anterior",
    "Dorsal fin tip",
    "Dorsal fin base — posterior",
    "Caudal peduncle — top",
    "Caudal peduncle — bottom",
    "Caudal fin tip — upper",
    "Caudal fin tip — lower",
    "Anal fin base — anterior",
    "Anal fin tip",
    "Anal fin base — posterior",
    "Caudal Fin Center",
]

FLIP_PAIRS = [
    (2, 4),   # dorsal fin base anterior <-> posterior
    (5, 6),   # caudal peduncle top <-> bottom
    (7, 8),   # caudal fin tip upper <-> lower
    (9, 11),  # anal fin base anterior <-> posterior
]

label_to_id = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
NUM_KP = len(KEYPOINT_NAMES)

# ── STEP 1: PARSE & CLEAN XML ─────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Parsing and cleaning annotations.xml")

tree = ET.parse(INPUT_XML)
root = tree.getroot()

clean_records = []
skipped = []

for img_elem in root.findall('image'):
    raw_name = img_elem.get('name')
    filename = os.path.basename(raw_name.replace('\\', '/'))
    width    = int(img_elem.get('width'))
    height   = int(img_elem.get('height'))

    # --- Points ---
    seen_labels = {}
    for pt in img_elem.findall('points'):
        label     = pt.get('label')
        pts_str   = pt.get('points', '')
        occluded  = int(pt.get('occluded', '0'))
        first_pt  = pts_str.split(';')[0]
        try:
            x, y = map(float, first_pt.split(','))
        except Exception:
            continue
        if label not in seen_labels:
            seen_labels[label] = (x, y, occluded)

    if len(seen_labels) == 0:
        skipped.append((filename, "zero keypoints"))
        continue

    # --- Bounding box (Fish_Box) ---
    box = None   # (x1, y1, x2, y2) in original pixels
    for box_elem in img_elem.findall('box'):
        if box_elem.get('label') == 'Fish_Box':
            xtl = float(box_elem.get('xtl'))
            ytl = float(box_elem.get('ytl'))
            xbr = float(box_elem.get('xbr'))
            ybr = float(box_elem.get('ybr'))
            box = (xtl, ytl, xbr, ybr)
            break

    # --- Build arrays ---
    coords     = np.zeros((NUM_KP, 2), dtype=np.float32)
    visibility = np.zeros(NUM_KP, dtype=np.int32)

    for label, (x, y, occ) in seen_labels.items():
        if label in label_to_id:
            idx = label_to_id[label]
            coords[idx] = [x, y]
            visibility[idx] = 1 if occ == 1 else 2

    clean_records.append({
        "filename":   filename,
        "width":      width,
        "height":     height,
        "coords":     coords,
        "visibility": visibility,
        "bbox":       box,
    })

print(f"  Valid images : {len(clean_records)}")
print(f"  Skipped      : {len(skipped)} -> {[s[0] for s in skipped]}")

# ── STEP 2: SPLIT ─────────────────────────────────────────────────────────────
print("\nSTEP 2: Splitting dataset")
random.seed(RANDOM_SEED)
random.shuffle(clean_records)

n = len(clean_records)
n_train = int(n * TRAIN_RATIO)
n_val   = int(n * VAL_RATIO)
n_test  = n - n_train - n_val

train_recs = clean_records[:n_train]
val_recs   = clean_records[n_train:n_train+n_val]
test_recs  = clean_records[n_train+n_val:]

print(f"  Total  : {n}")
print(f"  Train  : {len(train_recs)} (before augmentation)")
print(f"  Val    : {len(val_recs)}")
print(f"  Test   : {len(test_recs)}")

# ── STEP 3: AUGMENTATION ──────────────────────────────
print(f"\nSTEP 3: Augmenting training set x{AUG_PER_IMAGE}")

try:
    import albumentations as A
    import cv2
    HAS_ALB = True
except ImportError:
    HAS_ALB = False
    print("  [WARNING] albumentations not installed — skipping pixel augmentations")

if HAS_ALB:
    def build_aug_pipeline():
        return A.ReplayCompose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=15, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.25),
            A.Rotate(limit=12, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=(128,128,128)),
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.15, rotate_limit=0,
                               p=0.4, border_mode=cv2.BORDER_CONSTANT, value=(128,128,128)),
            A.RandomGamma(gamma_limit=(80,120), p=0.3),
            A.ImageCompression(quality_lower=75, quality_upper=100, p=0.2),
        ],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, check_each_transform=False),
        )

    aug_pipeline = build_aug_pipeline()

    def apply_flip_pairs(coords, flipped):
        if not flipped: return coords
        new = coords.copy()
        for i,j in FLIP_PAIRS:
            new[i], new[j] = coords[j].copy(), coords[i].copy()
        return new

    def apply_flip_pairs_vis(vis, flipped):
        if not flipped: return vis
        new = vis.copy()
        for i,j in FLIP_PAIRS:
            new[i], new[j] = vis[j], vis[i]
        return new

    def augment_record(rec, aug_idx, image_dir):
        img_path = os.path.join(image_dir, rec["filename"])
        if not os.path.exists(img_path): return None
        img = cv2.imread(img_path)
        if img is None: return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        coords     = rec["coords"].copy()
        visibility = rec["visibility"].copy()
        bbox       = rec.get("bbox")   # original bbox in original pixels

        kp_list = [(float(coords[k,0]), float(coords[k,1])) for k in range(NUM_KP)]

        result  = aug_pipeline(image=img_rgb, keypoints=kp_list)
        aug_img = result["image"]
        aug_kps = result["keypoints"]
        replay  = result["replay"]

        # Detect horizontal flip
        flipped = False
        for t in replay["transforms"]:
            if "HorizontalFlip" in t["__class_fullname__"]:
                flipped = True
                break

        new_coords = np.zeros((NUM_KP, 2), dtype=np.float32)
        new_vis    = visibility.copy()
        for k in range(NUM_KP):
            if k < len(aug_kps):
                new_coords[k] = [aug_kps[k][0], aug_kps[k][1]]
                x, y = new_coords[k]
                h, w = aug_img.shape[:2]
                if x < 0 or x >= w or y < 0 or y >= h:
                    new_vis[k] = 0
            else:
                new_vis[k] = 0

        new_coords = apply_flip_pairs(new_coords, flipped)
        new_vis    = apply_flip_pairs_vis(new_vis, flipped)

        # Resize to 224
        target = 224
        if aug_img.shape[0] != target or aug_img.shape[1] != target:
            orig_h, orig_w = aug_img.shape[:2]
            aug_img = cv2.resize(aug_img, (target, target))
            sx, sy = target/orig_w, target/orig_h
            new_coords[:,0] *= sx
            new_coords[:,1] *= sy
            if bbox is not None:
                bbox = (bbox[0]*sx, bbox[1]*sy, bbox[2]*sx, bbox[3]*sy)

        stem = Path(rec["filename"]).stem
        ext  = Path(rec["filename"]).suffix
        new_name = f"{stem}_aug{aug_idx:02d}{ext}"

        return {
            "filename":   new_name,
            "width":      target,
            "height":     target,
            "coords":     new_coords,
            "visibility": new_vis,
            "bbox":       bbox,
            "aug_image":  cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR),
        }
else:
    def augment_record(*args, **kwargs): return None

# ── STEP 4: OUTPUT DIRECTORIES 
print("\nSTEP 4: Creating output directories")
out_img_dir = os.path.join(OUTPUT_DIR, "images")
os.makedirs(out_img_dir, exist_ok=True)

# ── STEP 5: PROCESS SPLITS 
print("\nSTEP 5: Writing images and building records")

def copy_original(rec, image_dir, out_dir, split_prefix, target=224):
    src = os.path.join(image_dir, rec["filename"])
    if not os.path.exists(src): return None
    img = cv2.imread(src)
    if img is None: return None
    orig_h, orig_w = img.shape[:2]
    img_resized = cv2.resize(img, (target, target))

    new_name = f"{split_prefix}_{rec['filename']}"
    dst = os.path.join(out_dir, new_name)
    cv2.imwrite(dst, img_resized)

    sx, sy = target/orig_w, target/orig_h
    new_rec = copy.deepcopy(rec)
    new_rec["coords"][:,0] *= sx
    new_rec["coords"][:,1] *= sy
    if new_rec.get("bbox") is not None:
        x1,y1,x2,y2 = new_rec["bbox"]
        new_rec["bbox"] = (x1*sx, y1*sy, x2*sx, y2*sy)
    new_rec["width"] = target
    new_rec["height"] = target
    new_rec["filename"] = new_name
    return new_rec

all_final = {"train": [], "val": [], "test": []}
missing = []

for split, recs in [("val", val_recs), ("test", test_recs)]:
    print(f"  {split}: copying {len(recs)} originals...")
    for rec in recs:
        new_rec = copy_original(rec, IMAGE_DIR, out_img_dir, split)
        if new_rec:
            all_final[split].append(new_rec)
        else:
            missing.append(rec["filename"])

print(f"  train: originals + augmentations...")
aug_ok, aug_fail = 0, 0
for rec in train_recs:
    new_rec = copy_original(rec, IMAGE_DIR, out_img_dir, "train")
    if new_rec:
        all_final["train"].append(new_rec)
    else:
        missing.append(rec["filename"])
        aug_fail += 1
        continue

    for aug_idx in range(1, AUG_PER_IMAGE+1):
        aug_rec = augment_record(rec, aug_idx, IMAGE_DIR)
        if aug_rec is None:
            aug_fail += 1
            continue
        out_path = os.path.join(out_img_dir, f"train_{aug_rec['filename']}")
        cv2.imwrite(out_path, aug_rec["aug_image"])
        clean = {k:v for k,v in aug_rec.items() if k != "aug_image"}
        clean["filename"] = f"train_{clean['filename']}"
        all_final["train"].append(clean)
        aug_ok += 1

print(f"  Augmented images created : {aug_ok}")
print(f"  Failed                  : {aug_fail}")
if missing:
    print(f"  Missing source files    : {missing[:5]}...")

# ── STEP 6: WRITE COMBINED XML ────────────────────────────────────────────────
print("\nSTEP 6: Writing combined annotations XML")

def records_to_xml(records_dict):
    annots = ET.Element("annotations")
    ET.SubElement(annots, "version").text = "1.1"
    img_id = 0
    for split in ["train","val","test"]:
        for rec in records_dict[split]:
            img_elem = ET.SubElement(annots, "image",
                id=str(img_id),
                name=f"{split}/{rec['filename']}",
                width=str(rec["width"]),
                height=str(rec["height"]),
                subset=split)
            # Keypoints
            for k, name in enumerate(KEYPOINT_NAMES):
                if rec["visibility"][k] > 0:
                    x,y = rec["coords"][k]
                    ET.SubElement(img_elem, "points",
                        label=name,
                        source="manual" if "aug" not in rec["filename"] else "augmented",
                        occluded="1" if rec["visibility"][k]==1 else "0",
                        points=f"{x:.2f},{y:.2f}",
                        z_order="0")
            # Bounding box
            if rec.get("bbox") is not None:
                x1,y1,x2,y2 = rec["bbox"]
                ET.SubElement(img_elem, "box",
                    label="Fish_Box",
                    source="manual" if "aug" not in rec["filename"] else "augmented",
                    occluded="0",
                    xtl=f"{x1:.2f}", ytl=f"{y1:.2f}",
                    xbr=f"{x2:.2f}", ybr=f"{y2:.2f}",
                    z_order="0")
            img_id += 1
    return annots

xml_root = records_to_xml(all_final)
xml_tree = ET.ElementTree(xml_root)
ET.indent(xml_tree, space="  ")
out_xml = os.path.join(OUTPUT_DIR, "annotations_augmented.xml")
xml_tree.write(out_xml, encoding="utf-8", xml_declaration=True)
print(f"  Written: {out_xml}")

# ── STEP 7: WRITE SPLIT LISTS ─────────────────────────────────────────────────
print("\nSTEP 7: Writing split file lists")
for split, recs in all_final.items():
    list_path = os.path.join(OUTPUT_DIR, f"{split}_files.txt")
    with open(list_path, "w") as f:
        for rec in recs:
            f.write(rec["filename"] + "\n")
    print(f"  {split}: {len(recs)} files -> {list_path}")

print("\nDone!")