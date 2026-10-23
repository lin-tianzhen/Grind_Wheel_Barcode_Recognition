"""Global configuration for the grinding-wheel barcode project."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SYNTH_DIR = DATA_DIR / "synthetic"
TEST_DIR = DATA_DIR / "test"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TRAD_OUT = OUTPUT_DIR / "traditional"
DEEP_OUT = OUTPUT_DIR / "deep"
FIG_DIR = OUTPUT_DIR / "figures"

for _p in (MODEL_DIR, OUTPUT_DIR, TRAD_OUT, DEEP_OUT, FIG_DIR, TEST_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------
GT_TEXT = "4014855012527"
BARCODE_FORMAT = "EAN-13"
SEED = 2026

# The barcode is a dark label with white bars.  This is its bounding
# quadrilateral in RegulaImg2.bmp, measured once by hand.  It is only used by
# the *evaluation* code (never by the recognisers).
FULL_IMAGE = RAW_DIR / "RegulaImg2.bmp"
CROP_IMAGE = RAW_DIR / "BarAreaRoted.bmp"
GT_QUAD_FULL = [
    [1250, 2458],
    [1726, 2484],
    [1712, 2752],
    [1222, 2728],
]

# Region of the small, already cropped reference image that contains the bars
# (the human-readable digits below are deliberately excluded).
CROP_BAR_BOX = (20, 20, 505, 212)

# ---------------------------------------------------------------------------
# Pre-processing / data augmentation
# ---------------------------------------------------------------------------
SCENE_SIZE = 384                 # synthetic training scene size
TRAIN_PER_IMAGE = 260            # number of synthetic training scenes
VAL_PER_IMAGE = 60               # number of synthetic validation scenes
MAX_SIDE_LOCATE = 1280           # long side used by the traditional locator
DEEP_INFER_SIDE = 1024           # long side fed to the U-Net

# ---------------------------------------------------------------------------
# Deep model / training
# ---------------------------------------------------------------------------
BASE_CHANNELS = 16
EPOCHS = 45
BATCH_SIZE = 8
LEARNING_RATE = 1.5e-3
WEIGHT_DECAY = 1e-4
POS_WEIGHT = 2.0
NUM_WORKERS = 0
DEVICE = "auto"                   # auto -> mps, cuda, or cpu

MODEL_PATH = MODEL_DIR / "unet_barcode.pt"
TRAIN_HISTORY = MODEL_DIR / "train_history.json"
