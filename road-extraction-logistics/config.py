import os
import random
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
BACKBONE = "resnet50"  # "resnet50" (default) | "resnet101" (higher accuracy, ~2x params)
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
RDP_EPSILON = 2.0               # Ramer-Douglas-Peucker tolerance in pixels
PIXEL_RESOLUTION_METERS = 2.39  # DeepGlobe ~0.5 m/px at native res, scaled to 512px

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_train_val_split(data_dir=None, val_fraction=None, seed=None):
    """
    Return (train_files, val_files) as sorted lists of *_sat.jpg filenames.
    The split is deterministic for a given seed — both train.py and val.py
    call this to guarantee they see the same held-out images.
    """
    from src.dataset import RoadSegmentationDataset  # local import to avoid circular dep

    data_dir     = data_dir     or DATA_DIR
    val_fraction = val_fraction or VAL_SPLIT
    seed         = seed         or RANDOM_SEED

    all_files = sorted(f for f in os.listdir(data_dir) if f.endswith('_sat.jpg'))
    rng = random.Random(seed)
    shuffled = all_files[:]
    rng.shuffle(shuffled)

    split = int(len(shuffled) * (1 - val_fraction))
    return sorted(shuffled[:split]), sorted(shuffled[split:])
