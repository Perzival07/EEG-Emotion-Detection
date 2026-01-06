#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

# ---------------------------
# Hyper‑parameters
# ---------------------------

BATCH_SIZE = 4
 
PATCH_SIZE = 4
NUM_EPOCHS = 20  # allow many but we will early‑stop
EARLY_STOP_PAT = 5  # stop if val loss hasn't improved for this many epochs
LR_ENCODER = 2e-4
LR_PROJECTOR = 1e-4
WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 1.0
WARMUP_FRAC = 0.03
MAX_SAMPLES = None  # set to an int for quick experiments
RESULTS_FILE = "test_predictions.jsonl"
EMBED_DIM = 128
ENCODER_OUTPUT_DIM = EMBED_DIM
TRANSFORMER_INPUT_DIM = EMBED_DIM

# ---------------------------
# Performance Optimization Settings
# ---------------------------

# DataLoader optimizations
NUM_WORKERS = 4  # Number of worker processes for data loading (0 = single-threaded)
PIN_MEMORY = True  # Faster GPU transfer
PERSISTENT_WORKERS = True  # Keep workers alive between epochs

# Mixed precision training
USE_MIXED_PRECISION = True  # Enable automatic mixed precision (AMP)
MIXED_PRECISION_DTYPE = "bf16"  # "bf16" (better) or "fp16" (wider support)

# Gradient accumulation (allows larger effective batch size)
GRADIENT_ACCUMULATION_STEPS = 1  # Accumulate gradients over N steps before optimizer.step()

# Validation batch size (increase for faster validation)
VAL_BATCH_SIZE = 4  # Increase from 1 for faster validation

# Model compilation (PyTorch 2.0+)
USE_TORCH_COMPILE = False  # Enable torch.compile for faster inference (experimental)

# Inference optimizations
INFERENCE_BATCH_SIZE = 8  # Batch size for inference
INFERENCE_NUM_WORKERS = 2  # Workers for inference data loading
