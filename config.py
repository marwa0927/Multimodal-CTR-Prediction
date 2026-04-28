from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR        = ROOT / "data"
OUTPUTS_DIR     = ROOT / "outputs"
FEATURES_DIR    = OUTPUTS_DIR / "features"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"

SEED = 42

# Task 1 — tag encoder hyperparameters
TAG_VOCAB_SIZE = 11_740
TAG_EMB_DIM    = 64
ITEM_EMB_DIM   = 128
MAX_TAGS       = 64          # pad/truncate tag lists to this length

T1_EPOCHS      = 30
T1_LR          = 1e-3
T1_BATCH_SIZE  = 4096
T1_BLEND_ALPHA = 0.5         # weight on tag-model output in final blend
