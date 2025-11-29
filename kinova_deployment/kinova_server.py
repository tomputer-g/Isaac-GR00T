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
Kinova Policy Server - Runs GR00T policy inference on GPU machine

This script starts a server that:
1. Loads the GR00T policy checkpoint
2. Listens for observation requests from the robot client
3. Runs policy inference
4. Returns actions to the client

Usage:
    python kinova_server.py --checkpoint train_result/checkpoint-5000 --port 5555
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gr00t.data.dataset import ModalityConfig
from gr00t.data.schema import EmbodimentTag
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.eval.robot import RobotInferenceServer
from gr00t.model.policy import Gr00tPolicy
from gr00t.model.transforms import GR00TTransform


def create_kinova_policy(checkpoint_path: str, device: str = "cuda:0") -> Gr00tPolicy:
    """Create and load Kinova policy from checkpoint."""
    
    print(f"Loading Kinova GR00T policy from: {checkpoint_path}")
    
    # Define modality configuration for Kinova
    video_keys = ["video.external"]#, "video.wrist"]
    state_keys = ["state.arm_joints", "state.gripper"]
    action_keys = ["action.arm_joints", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    
    observation_indices = [0]
    action_indices = range(16)
    
    modality_config = {
        "video": ModalityConfig(
            delta_indices=observation_indices,
            modality_keys=video_keys,
        ),
        "state": ModalityConfig(
            delta_indices=observation_indices,
            modality_keys=state_keys,
        ),
        "action": ModalityConfig(
            delta_indices=action_indices,
            modality_keys=action_keys,
        ),
        "language": ModalityConfig(
            delta_indices=observation_indices,
            modality_keys=language_keys,
        )
    }
    
    # Define transforms
    transforms = [
        # Video transforms
        VideoToTensor(apply_to=video_keys),
        VideoCrop(apply_to=video_keys, scale=0.95),
        VideoResize(apply_to=video_keys, height=224, width=224, interpolation="linear"),
        VideoColorJitter(
            apply_to=video_keys,
            brightness=0.3,
            contrast=0.4,
            saturation=0.5,
            hue=0.08,
        ),
        VideoToNumpy(apply_to=video_keys),
        # State transforms
        StateActionToTensor(apply_to=state_keys),
        StateActionTransform(
            apply_to=state_keys,
            normalization_modes={
                "state.arm_joints": "min_max",
                "state.gripper": "min_max",
            }
        ),
        # Action transforms
        StateActionToTensor(apply_to=action_keys),
        StateActionTransform(
            apply_to=action_keys,
            normalization_modes={
                "action.arm_joints": "min_max",
                "action.gripper": "min_max",
            }
        ),
        # Concat transforms
        ConcatTransform(
            video_concat_order=video_keys,
            state_concat_order=state_keys,
            action_concat_order=action_keys,
        ),
        GR00TTransform(
            state_horizon=len(observation_indices),
            action_horizon=len(action_indices),
            max_state_dim=64,
            max_action_dim=32,
        ),
    ]
    
    composed_transform = ComposedModalityTransform(transforms=transforms)
    
    # Load policy
    policy = Gr00tPolicy(
        model_path=checkpoint_path,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        modality_config=modality_config,
        modality_transform=composed_transform,
        device=device,
    )
    
    print(f"✓ Policy loaded successfully on {device}")
    print(f"  Modality config: {list(modality_config.keys())}")
    print(f"  Action horizon: {len(action_indices)}")
    
    return policy


def main():
    parser = argparse.ArgumentParser(description="Kinova GR00T Policy Server")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to policy checkpoint (e.g., train_result/checkpoint-5000)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="Port to run server on (default: 5555)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="*",
        help="Host address to bind to (default: * for all interfaces)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on (default: cuda:0)"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("Kinova GR00T Policy Server")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Port: {args.port}")
    print(f"Device: {args.device}")
    print("=" * 70)
    
    # Create policy
    policy = create_kinova_policy(args.checkpoint, args.device)
    
    # Start server
    print(f"\nStarting inference server...")
    print(f"Clients can connect to: tcp://<this-machine-ip>:{args.port}")
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    
    RobotInferenceServer.start_server(policy, port=args.port)


if __name__ == "__main__":
    main()
