import torch

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
SAVE_EVERY_N_EPOCHS = 1
VAL_SPLIT = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 0        # set >0 on Linux/Mac; Windows multiprocessing requires __main__ guard
PIN_MEMORY = True

# ── Model ─────────────────────────────────────────────────────────────────────
BACKBONE = "resnet50"  # "resnet50" | "resnet101" | "mobilenet_v3_large"
NUM_CLASSES = 2

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "data/raw/train"
CHECKPOINT_PATH = "models/checkpoint.pth"
FINAL_MODEL_PATH = "models/checkpoints/DeeplabsV3_road_final.pth"
INPUT_DIR = "data/raw/test"
OUTPUT_DIR = "data/masks/predicted"
CLEAN_DIR = "data/masks/cleaned"
SKEL_DIR = "data/masks/skeletons"
MASK_DIR = "data/masks/predicted"
PATH_OUTPUT_DIR = "data/paths"

# ── Inference ─────────────────────────────────────────────────────────────────
THRESHOLD = 0.5
USE_CLAHE = True

# ── Pathfinding ───────────────────────────────────────────────────────────────
USE_SKELETON = True
RDP_EPSILON = 2.0
PIXEL_RESOLUTION_METERS = 2.39  # DeepGlobe ~0.5m/px native, scaled to 512px

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
