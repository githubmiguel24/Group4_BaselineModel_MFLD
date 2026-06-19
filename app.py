import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import os

import config
from model import MFLDNet

# ── Page & CSS ──────────────────────────────────────────────
st.set_page_config(page_title="BettaTool", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        [data-testid="collapsedControl"] { display: none; }
        .stApp { background-color: #FFFFFF; }
        [data-testid="block-container"] { padding-top: 2rem; padding-bottom: 2rem; max-width: 90%; }
        .main-brand {
            font-size: 38px; font-weight: 800; color: #000000; text-align: center;
            padding: 20px; background: linear-gradient(180deg, #E3F2FD 0%, #64B5F6 100%);
            border: 2px solid #000000; border-radius: 8px; margin-bottom: 20px; letter-spacing: 1px;
            box-shadow: 2px 2px 0px #000000;
        }
        .header-title {
            font-size: 24px; font-weight: 600; color: #1976D2; margin-top: 10px;
            margin-bottom: 15px; padding-bottom: 5px; border-bottom: 2px solid #000000;
        }
        .legend-container {
            display: flex; flex-wrap: wrap; gap: 10px; font-size: 13px; margin-top: 30px;
            color: #000000; background-color: #FFFFFF; padding: 20px; border-radius: 8px;
            border: 2px solid #000000;
        }
        .legend-item {
            display: flex; align-items: center; width: 23%; min-width: 160px; font-weight: 500;
        }
        .color-dot {
            width: 12px; height: 12px; border-radius: 50%; margin-right: 8px;
            display: inline-block; border: 1px solid #000000;
        }
    </style>
""", unsafe_allow_html=True)

# ── 13 keypoints ────────────────
NUM_KP = 13
KP_NAMES = [
    "Snout", "Eye", "Nape", "Dorsal Fin Base",
    "Caudal Peduncle (Top)", "Caudal Peduncle (Bottom)",
    "Anal Fin Base", "Pelvic Fin Base", "Pectoral Fin Base",
    "Gill Cover", "Mouth Corner", "Operculum Edge", "Body Centre"
]
COLORS_HEX = [
    "#808080", "#FF0000", "#FF69B4", "#800080", "#008000",
    "#006400", "#FFA500", "#FFFF00", "#32CD32", "#FF00FF",
    "#00FF00", "#90EE90", "#00FFFF"
]
COLORS_BGR = [tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0,2,4))[::-1] for h in COLORS_HEX]

# ── Model ──────────────────────────────
@st.cache_resource
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use config.NUM_KEYPOINTS as the model was originally trained.
    # It might be >13 – that's okay, we'll slice the heatmaps later.
    model = MFLDNet(
        num_keypoints=config.NUM_KEYPOINTS,   # keep the original!
        embed_dim=config.EMBED_DIM,
        num_blocks=12,
        kernel_size=config.KERNEL_SIZE,
        dropout_rate=0.0,
        heatmap_size=config.HEATMAP_SIZE,
    ).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        st.error(f"Weights '{model_path}' not found.")
    model.eval()
    return model, device

model, device = load_model("best.pth")

# ── UI ───────────────────────────
st.markdown('<div class="main-brand">BettaTool</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image to Analyze", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.markdown('<div class="header-title">Keypoint Analysis</div>', unsafe_allow_html=True)

    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    orig_w, orig_h = image.size
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Preprocess
    resized = cv2.resize(img_cv, (config.INPUT_SIZE, config.INPUT_SIZE))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm_img = (resized.astype(np.float32) / 255.0 - mean) / std
    input_tensor = torch.from_numpy(norm_img.transpose(2,0,1)).float().unsqueeze(0).to(device)

    # Inference
    with st.spinner("Analyzing morphology..."):
        with torch.no_grad():
            heatmaps, _ = model(input_tensor)   # shape: (1, C, H, W)

    total_channels = heatmaps.shape[1]
    if total_channels < NUM_KP:
        st.error(f"Model only has {total_channels} heatmap channels, but we need {NUM_KP}.")
        st.stop()

    # Slice to first 13
    heatmaps_13 = heatmaps[:, :NUM_KP, :, :]   # (1, 13, H, W)
    heatmaps_np = heatmaps_13.cpu().numpy()[0] # (13, H, W)
    heatmap_h, heatmap_w = heatmaps_np.shape[1], heatmaps_np.shape[2]

    coords = np.zeros((NUM_KP, 2), dtype=np.float32)
    for i in range(NUM_KP):
        hm = heatmaps_np[i]
        max_idx = np.unravel_index(np.argmax(hm), hm.shape)  # (y, x)
        coords[i, 1] = max_idx[0] / (heatmap_h - 1)  # y
        coords[i, 0] = max_idx[1] / (heatmap_w - 1)  # x

    # Scale to original image size
    coords[:, 0] *= orig_w
    coords[:, 1] *= orig_h

    # ── Draw exactly 13 keypoints ─────────────────────────────
    for i in range(NUM_KP):
        x, y = coords[i]
        cv2.circle(img_cv, (int(x), int(y)), 6, COLORS_BGR[i], -1)
        cv2.circle(img_cv, (int(x), int(y)), 7, (0,0,0), 1)

    result_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    # Display
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(result_rgb, use_container_width=True)

    # Legend (always 13 items)
    legend_html = '<div class="legend-container">'
    for i, name in enumerate(KP_NAMES):
        legend_html += f'<div class="legend-item"><span class="color-dot" style="background-color: {COLORS_HEX[i]};"></span>{name}</div>'
    legend_html += '</div>'
    st.markdown(legend_html, unsafe_allow_html=True)

else:
    st.info("Upload a betta‑fish image to process.")