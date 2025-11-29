#!/usr/bin/env python3
"""
Check joint limits from training dataset.

This script analyzes your training data to find the actual joint ranges
and helps you set appropriate JOINT_LIMITS in config.py.

Usage:
    python check_training_limits.py --dataset datasets/visible+bowl_36eps
"""

import argparse
import json
from pathlib import Path

import numpy as np


def check_joint_limits(dataset_path: str):
    """Check joint limits from training dataset stats."""
    
    dataset_path = Path(dataset_path)
    stats_path = dataset_path / "meta" / "stats.json"
    
    if not stats_path.exists():
        print(f"❌ Stats file not found: {stats_path}")
        print(f"   Make sure the dataset path is correct")
        return
    
    print("=" * 70)
    print("Training Dataset Joint Limits Analysis")
    print("=" * 70)
    print(f"Dataset: {dataset_path}")
    print(f"Stats file: {stats_path}")
    print()
    
    # Load stats
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    # Check state statistics (joint positions during training)
    state_keys = [
        "state.arm_joints",
        "state.gripper"
    ]
    
    print("📊 Joint Statistics from Training Data:")
    print("-" * 70)
    
    for key in state_keys:
        if key in stats:
            stat = stats[key]
            min_vals = stat.get('min', [])
            max_vals = stat.get('max', [])
            mean_vals = stat.get('mean', [])
            
            if key == "state.arm_joints":
                joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
                print("\n🔧 Arm Joints:")
                for i, name in enumerate(joint_names):
                    if i < len(min_vals):
                        min_val = min_vals[i]
                        max_val = max_vals[i]
                        mean_val = mean_vals[i]
                        range_val = max_val - min_val
                        print(f"  {name:>10}: min={min_val:6.1f}°  max={max_val:6.1f}°  mean={mean_val:6.1f}°  range={range_val:5.1f}°")
            
            elif key == "state.gripper":
                if len(min_vals) > 0:
                    print(f"\n✋ Gripper:")
                    print(f"  {'gripper':>10}: min={min_vals[0]:6.1f}   max={max_vals[0]:6.1f}   mean={mean_vals[0]:6.1f}")
    
    # Generate recommended config
    print("\n" + "=" * 70)
    print("📝 Recommended JOINT_LIMITS for config.py:")
    print("=" * 70)
    print()
    print("JOINT_LIMITS = {")
    
    if "state.arm_joints" in stats:
        stat = stats["state.arm_joints"]
        min_vals = stat.get('min', [])
        max_vals = stat.get('max', [])
        joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        
        for i, name in enumerate(joint_names):
            if i < len(min_vals):
                min_val = max(0.0, min_vals[i] - 20.0)  # Add 20° buffer below
                max_val = min(360.0, max_vals[i] + 20.0)  # Add 20° buffer above
                print(f"    '{name}': ({min_val:.1f}, {max_val:.1f}),")
    
    if "state.gripper" in stats:
        stat = stats["state.gripper"]
        min_vals = stat.get('min', [])
        max_vals = stat.get('max', [])
        if len(min_vals) > 0:
            min_val = max(0.0, min_vals[0] - 5.0)
            max_val = min(100.0, max_vals[0] + 5.0)
            print(f"    'gripper': ({min_val:.1f}, {max_val:.1f})")
    
    print("}")
    print()
    print("⚠️  Note: Buffers of ±20° (joints) and ±5 (gripper) added for safety")
    print("   Adjust based on your robot's actual physical limits")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check training data joint limits")
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/visible+bowl_36eps",
        help="Path to training dataset"
    )
    args = parser.parse_args()
    
    check_joint_limits(args.dataset)
