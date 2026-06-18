import os

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(ROOT_DIR, "data")
IMAGE_DIR      = os.path.join(DATA_DIR, "augmented", "images")
ANNOT_FILE     = os.path.join(DATA_DIR, "augmented", "annotations_augmented.xml")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
LOG_DIR        = os.path.join(ROOT_DIR, "logs")
OUTPUT_DIR     = os.path.join(ROOT_DIR, "outputs")

for _d in (CHECKPOINT_DIR, LOG_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)

# ── Dataset ──────────────────────────────────────────────────────────────────
NUM_KEYPOINTS   = 13          # Ventral Base & Ventral Tip removed
INPUT_SIZE      = 224
HEATMAP_SIZE    = 56
GAUSSIAN_SIGMA  = 2.0

TRAIN_VAL_RATIO = 0.80
TRAIN_SPLIT     = 0.70
VAL_SPLIT       = 0.30
TEST_RATIO      = 0.20
RANDOM_SEED     = 42

# ── MFLD-net architecture ────────────────────────────────────────────────────
EMBED_DIM       = 512
NUM_CONV_BLOCKS = 8           
KERNEL_SIZE     = 9
DROPOUT_RATE    = 0.20
PATCH_SIZE      = 4

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE      = 48
NUM_EPOCHS      = 25           
LEARNING_RATE   = 5e-4
LR_DECAY_STEP   = 10
LR_DECAY_GAMMA  = 0.3
WEIGHT_DECAY    = 1e-4
ADAM_BETA1      = 0.9
ADAM_BETA2      = 0.999
ADAM_EPS        = 1e-8
MULTITASK_ALPHA = 0.5

# ── Augmentation (offline only) ──────────────────────────────────────────────
AUG_H_FLIP_PROB     = 0.5
AUG_SHIFT_LIMIT     = 0.0625
AUG_SCALE_LIMIT     = 0.20
AUG_SHIFT_SCALE_PROB= 0.5
AUG_ROTATE_LIMIT    = 20
AUG_ROTATE_PROB     = 0.5
AUG_BLUR_LIMIT      = 1
AUG_BLUR_PROB       = 0.3
AUG_RGB_SHIFT_LIMIT = 25
AUG_RGB_SHIFT_PROB  = 0.3

# ── Evaluation ───────────────────────────────────────────────────────────────
OKS_SIGMAS = None
OKS_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# ── Keypoint names (order must match augment_and_split.py) ──────────────────
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

# Keypoint groups for downstream morphometry
CAUDAL_SPREAD_KPS      = (7, 12, 8)        # 7,12,8
ANAL_FIN_LENGTH_KPS    = (9, 10, 11)       # 9,10,11
DORSAL_FIN_LENGTH_KPS  = (2, 3, 4)         # 2,3,4
BODY_LENGTH_KPS        = (0, 5, 6)         # 0,5,6