# Hyperparameters and paths for reproducibility
import torch

# Data paths (update with your root dir scanning logic)
DATA_ROOT = 'data/'  # Root folder with 1000 subfolders
# Example: Use os.walk or glob to build data_paths list in train.py/eval.py

# Model and training params
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
ACCUM_STEPS = 5
PATIENCE = 2
MIN_DELTA = 0.001

NUM_WORKERS = 4
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MAX_POINTS = 1024
CROP_SIZE = (224, 224)

# Splits (fractions)
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
# Test is remainder
