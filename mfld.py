import argparse
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

import config
from model import MFLDNet

# Default folders
AUGMENTED_IMAGE_DIR = "data/augmented/images"
DEFAULT_MODEL = "overfit_model.pth"

def list_test_images():
    test_list_path = "data/augmented/test_files.txt"
    if not os.path.exists(test_list_path):
        print("Test file list not found at", test_list_path)
        return
    with open(test_list_path) as f:
        test_files = [line.strip() for line in f if line.strip()]
    print(f"Test images available ({len(test_files)} total):")
    for fn in test_files[:10]:
        print(f"  {fn}")
    if len(test_files) > 10:
        print(f"  ... and {len(test_files)-10} more")

def resolve_image_path(image_path):
    if os.path.exists(image_path):
        return image_path
    # Try adding the augmented folder prefix
    candidate = os.path.join(AUGMENTED_IMAGE_DIR, image_path)
    if os.path.exists(candidate):
        return candidate
    # If still not found, return the original (will trigger a proper error later)
    return image_path

def run_inference(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on: {device}")

    # 1. Load model with the EXACT architecture you trained
    model = MFLDNet(
        num_keypoints=config.NUM_KEYPOINTS,   # 13
        embed_dim=512,
        num_blocks=12,
        kernel_size=config.KERNEL_SIZE,
        dropout_rate=0.0,
        heatmap_size=config.HEATMAP_SIZE,
    ).to(device)

    # 2. Load the raw state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("[*] Model loaded.")

    # 3. Load and preprocess the image
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb_img.shape[:2]

    # Resize to 224x224
    resized = cv2.resize(rgb_img, (config.INPUT_SIZE, config.INPUT_SIZE))
    # Normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm_img = (resized.astype(np.float32) / 255.0 - mean) / std

    # To tensor (1, 3, 224, 224)
    input_tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    # 4. Forward pass
    with torch.no_grad():
        heatmaps, coords = model(input_tensor)

    # 5. Post‑process: scale [0,1] coords back to original image size
    pred_coords = coords[0].cpu().numpy()
    pred_coords[:, 0] *= orig_w
    pred_coords[:, 1] *= orig_h

    # 6. Visualization
    plt.figure(figsize=(12, 8))
    plt.imshow(rgb_img)
    plt.scatter(pred_coords[:, 0], pred_coords[:, 1],
                c='cyan', s=60, edgecolors='white', linewidths=1.5,
                label='MFLD-net Predictions')

    # Label points with names
    for i, (x, y) in enumerate(pred_coords):
        plt.text(x + 8, y + 8, config.KEYPOINT_NAMES[i],
                 color='yellow', fontsize=8,
                 bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    plt.title("MFLD-net Keypoint Detection")
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MFLD-net inference on a single image")
    parser.add_argument("--image", type=str, required=True,
                        help="Path or filename of the image (looks in data/augmented/images/ if not found)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Path to the trained model (default: {DEFAULT_MODEL})")
    parser.add_argument("--list", action="store_true",
                        help="List test images available and exit")
    args = parser.parse_args()

    if args.list:
        list_test_images()
    else:
        full_image_path = resolve_image_path(args.image)
        print(f"[*] Using image: {full_image_path}")
        run_inference(full_image_path, args.model)