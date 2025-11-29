# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration for Kinova Gen3 deployment."""

import os
from pathlib import Path

# ============================================================================
# Robot Connection Settings
# ============================================================================
ROBOT_IP = "192.168.2.9"  
ROBOT_PORT = 10000
ROBOT_USERNAME = "admin" # this is admin and admin in the kinova_arm.py file 
ROBOT_PASSWORD = "admin"

# ============================================================================
# Camera Settings
# ============================================================================
EXTERNAL_CAMERA_INDEX = 0  
WRIST_CAMERA_INDEX = 1   
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ============================================================================
# Control Settings
# ============================================================================
CONTROL_FREQUENCY = 30  # Hz - should match training data FPS
ACTION_HORIZON = 16     # Number of action steps to execute per inference
MAX_EPISODE_STEPS = 500 # Maximum steps per episode

# Joint limits (in degrees) - Based on actual training data
# From datasets/visible+bowl_36eps/meta/stats.json
# Format: (min - buffer, max + buffer) with safety margins
JOINT_LIMITS = {
    'joint_1': (0.0, 360.0),      # Training: 0.0-359.9 (full rotation)
    'joint_2': (0.0, 360.0),      # Training: 0.1-359.7 (full rotation)
    'joint_3': (192.0, 328.0),    # Training: 213.0-308.0 + buffer
    'joint_4': (0.0, 360.0),      # Training: 0.0-360.0 (full rotation) - FIXED
    'joint_5': (264.0, 360.0),    # Training: 284.2-349.9 + buffer
    'joint_6': (60.0, 200.0),     # Training: 80.9-178.2 + buffer
    'gripper': (0.0, 100.0)       # Training: 0.9-99.6
}

# Safety margins (degrees) - reduce range for safety
# Full rotation joints (1,2,4) use smaller margin, limited joints use 10°
SAFETY_MARGIN = 2.0  # Reduced for full-rotation joints

# Maximum velocity/acceleration limits
MAX_JOINT_VELOCITY = 20.0  # degrees/sec
MAX_JOINT_ACCELERATION = 50.0  # degrees/sec^2

# ============================================================================
# Model Settings
# ============================================================================
REPO_PATH = Path(__file__).parent.parent
DEFAULT_CHECKPOINT = REPO_PATH / "train_result" / "checkpoint-5000"
# DEFAULT_CHECKPOINT = "/home/ishita/L3D_Team4_src/Isaac-GR00T/train_result/checkpoint-5000"
# DATA_PATH = REPO_PATH / "datasets" / "merged_dataset_nov22_30eps" 
DATA_PATH = REPO_PATH / "datasets" / "visible+bowl_36eps" 

# Normalization statistics (from training)
STATS_PATH = DATA_PATH / "meta" / "stats.json"

# ============================================================================
# Task Settings
# ============================================================================
DEFAULT_TASK = "Pick up the orange cup and place it on the black cross"

# ============================================================================
# Visualization & Logging
# ============================================================================
ENABLE_VISUALIZATION = True
LOG_DIR = REPO_PATH / "kinova_deployment" / "logs"
SAVE_VIDEOS = False
VIDEO_DIR = REPO_PATH / "kinova_deployment" / "videos"

# ============================================================================
# Device Settings
# ============================================================================
DEVICE = "cuda:0" 
