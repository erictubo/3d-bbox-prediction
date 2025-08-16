"""
Device, data paths, and hyperparameters for reproducibility
"""

import torch

# DEVICE
NUM_WORKERS = 4
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # for Apple Silicon

# DATA PATHS
# Raw data
DATA_ROOT = 'data/'
# Splits
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
# Test is remainder

# Processing
MAX_POINTS = 1024       # for point cloud
CROP_SIZE = (224, 224)  # for rgb crop

# HYPERPARAMETERS
# Training
NUM_EPOCHS = 10
BATCH_SIZE = 16
ACCUM_STEPS = 5
# Optimizer
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
# Early stopping
PATIENCE = 3
MIN_DELTA = 0.001