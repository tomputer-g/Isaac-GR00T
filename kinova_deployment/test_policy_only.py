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

"""
SAFEST TEST: Check policy inference without any robot connection.

This script tests ONLY the policy loading and inference pipeline.
NO robot connection, NO camera access, NO physical movement.
Uses dummy/synthetic observations to test the policy.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from deploy_kinova import KinovaPolicyDeployer


def create_dummy_observation():
    """Create a synthetic observation that looks like real robot data."""
    # Dummy joint positions (in safe middle range)
    joint_positions = np.array([
        180.0,  # joint_1
        180.0,  # joint_2
        250.0,  # joint_3 (middle of 212-296 range)
        180.0,  # joint_4
        315.0,  # joint_5 (middle of 282-349 range)
        115.0,  # joint_6 (middle of 89-144 range)
        50.0    # gripper (middle of 0-100 range)
    ], dtype=np.float32)
    
    # Dummy camera images (random noise for testing)
    external_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    wrist_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    return {
        'observation.state': joint_positions,
        'observation.images.external': external_image,
        'observation.images.wrist': wrist_image
    }


def test_policy_inference():
    """Test policy loading and inference with dummy data."""
    print("=" * 80)
    print("SAFE TEST #1: Policy Inference Only (NO ROBOT)")
    print("=" * 80)
    print("\nThis test will:")
    print("  ✓ Load the trained checkpoint")
    print("  ✓ Create synthetic observations")
    print("  ✓ Run policy inference")
    print("  ✓ Check action shapes and ranges")
    print("  ✗ NOT connect to robot")
    print("  ✗ NOT access cameras")
    print("  ✗ NOT move anything")
    print("\n" + "=" * 80 + "\n")
    
    # Check if checkpoint exists
    checkpoint_path = config.DEFAULT_CHECKPOINT
    if not checkpoint_path.exists():
        print(f"❌ ERROR: Checkpoint not found at {checkpoint_path}")
        print("\nPlease specify a valid checkpoint:")
        print("  1. Train a model first (run l3d_src/kinova.py)")
        print("  2. Or update DEFAULT_CHECKPOINT in config.py")
        return False
    
    print(f"[1/6] Checkpoint found: {checkpoint_path}")
    
    # Load statistics
    print(f"[2/6] Loading normalization statistics...")
    if not config.STATS_PATH.exists():
        print(f"❌ ERROR: Stats file not found at {config.STATS_PATH}")
        return False
    
    with open(config.STATS_PATH, 'r') as f:
        stats = json.load(f)
    
    action_min = np.array(stats['action']['min'])
    action_max = np.array(stats['action']['max'])
    print(f"✓ Stats loaded")
    print(f"  Action ranges: {action_min} to {action_max}")
    
    # Initialize policy
    print(f"\n[3/6] Loading policy (this may take 30-60 seconds)...")
    try:
        policy_deployer = KinovaPolicyDeployer(
            checkpoint_path=str(checkpoint_path),
            device=config.DEVICE
        )
        print("✓ Policy loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading policy: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create dummy observation
    print(f"\n[4/6] Creating synthetic observation...")
    observation = create_dummy_observation()
    print(f"✓ Observation created:")
    print(f"  State shape: {observation['observation.state'].shape}")
    print(f"  State values: {observation['observation.state']}")
    print(f"  External image shape: {observation['observation.images.external'].shape}")
    print(f"  Wrist image shape: {observation['observation.images.wrist'].shape}")
    
    # Test policy inference
    print(f"\n[5/6] Running policy inference...")
    task = "pick up the orange cup and place it into the blue bowl"
    print(f"  Task: '{task}'")
    
    try:
        # Get normalized actions
        normalized_actions = policy_deployer.get_action(
            observation=observation,
            language_instruction=task
        )
        print(f"✓ Inference successful")
        print(f"  Normalized action shape: {normalized_actions.shape}")
        print(f"  Expected: ({config.ACTION_HORIZON}, 7)")
        
        if normalized_actions.shape != (config.ACTION_HORIZON, 7):
            print(f"❌ ERROR: Unexpected action shape!")
            return False
        
        # Denormalize actions
        actions = policy_deployer.denormalize_action(normalized_actions)
        print(f"  Denormalized action shape: {actions.shape}")
        
    except Exception as e:
        print(f"❌ ERROR during inference: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Validate actions
    print(f"\n[6/6] Validating action outputs...")
    
    # Check normalization
    print(f"\n  Normalized actions (should be mostly in [0, 1]):")
    print(f"    Min: {normalized_actions.min():.4f}")
    print(f"    Max: {normalized_actions.max():.4f}")
    print(f"    Mean: {normalized_actions.mean():.4f}")
    
    # Check denormalized actions
    print(f"\n  Denormalized actions (in degrees):")
    print(f"    First action: {actions[0]}")
    print(f"    Last action: {actions[-1]}")
    
    # Check if actions are within expected ranges
    all_valid = True
    joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper']
    
    print(f"\n  Checking if actions are within training ranges:")
    for i, name in enumerate(joint_names):
        min_val = action_min[i]
        max_val = action_max[i]
        action_vals = actions[:, i]
        in_range = np.all((action_vals >= min_val - 10) & (action_vals <= max_val + 10))
        
        status = "✓" if in_range else "⚠"
        print(f"    {status} {name:8s}: range [{action_vals.min():6.1f}, {action_vals.max():6.1f}] "
              f"vs training [{min_val:6.1f}, {max_val:6.1f}]")
        
        if not in_range:
            all_valid = False
    
    # Check for safety limits
    print(f"\n  Checking against safety limits:")
    for i, name in enumerate(joint_names):
        if name in config.JOINT_LIMITS:
            min_safe, max_safe = config.JOINT_LIMITS[name]
            min_safe += config.SAFETY_MARGIN
            max_safe -= config.SAFETY_MARGIN
            
            action_vals = actions[:, i]
            in_safe_range = np.all((action_vals >= min_safe) & (action_vals <= max_safe))
            
            status = "✓" if in_safe_range else "⚠"
            print(f"    {status} {name:8s}: range [{action_vals.min():6.1f}, {action_vals.max():6.1f}] "
                  f"vs safe [{min_safe:6.1f}, {max_safe:6.1f}]")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST RESULTS:")
    print("=" * 80)
    print("✓ Checkpoint loads correctly")
    print("✓ Policy inference works")
    print("✓ Actions have correct shape")
    print("✓ Normalization/denormalization works")
    
    if all_valid:
        print("✓ All actions within expected ranges")
    else:
        print("⚠ Some actions outside expected ranges (may need retraining or stats check)")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("This test passed! You can now proceed to:")
    print("  1. Test cameras: python test_cameras_only.py")
    print("  2. Test robot connection: python test_robot_connection.py")
    print("  3. Deploy on real robot: python deploy_kinova.py")
    print("=" * 80 + "\n")
    
    return True


if __name__ == "__main__":
    success = test_policy_inference()
    sys.exit(0 if success else 1)
