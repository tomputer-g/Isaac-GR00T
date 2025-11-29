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
Deploy trained GR00T model on Kinova Gen3 robotic arm.

This script loads a trained checkpoint and runs closed-loop control on the physical robot.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import gr00t
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
from gr00t.model.policy import Gr00tPolicy
from gr00t.model.transforms import GR00TTransform

import config
from kinova_robot import KinovaGen3Robot

class KinovaPolicyDeployer:
    """Handles policy inference and preprocessing for Kinova deployment."""
    
    def __init__(self, checkpoint_path: str, device: str = config.DEVICE):
        """
        Initialize policy deployer.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on (e.g., 'cuda:0' or 'cpu')
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        
        # Load normalization statistics
        self.stats = self._load_stats()
        
        # Setup modality configuration (same as training)
        self.modality_config = self._get_modality_config()
        
        # Setup transforms (same as training)
        self.transforms = self._get_transforms()
        
        # Load policy
        print(f"Loading policy from {checkpoint_path}...")
        self.policy = Gr00tPolicy(
            model_path=checkpoint_path,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            modality_config=self.modality_config,
            modality_transform=self.transforms,
            device=device,
        )
        print("Policy loaded")
    
    def _load_stats(self) -> dict:
        """Load normalization statistics from training data."""
        stats_path = config.STATS_PATH
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats file not found: {stats_path}")
        
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        return stats
    
    def _get_modality_config(self) -> dict:
        """Get modality configuration (same as training)."""
        video_keys = ["video.external", "video.wrist"]
        state_keys = ["state.arm_joints", "state.gripper"]
        action_keys = ["action.arm_joints", "action.gripper"]
        language_keys = ["annotation.human.task_description"]
        
        observation_indices = [0]
        action_indices = list(range(config.ACTION_HORIZON))
        
        return {
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
    
    def _get_transforms(self) -> ComposedModalityTransform:
        """Get data transforms (same as training)."""
        video_keys = ["video.external", "video.wrist"]
        state_keys = ["state.arm_joints", "state.gripper"]
        action_keys = ["action.arm_joints", "action.gripper"]
        
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
                state_horizon=1,
                action_horizon=config.ACTION_HORIZON,
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        
        return ComposedModalityTransform(transforms=transforms)
    
    def preprocess_observation(self, observation: dict) -> dict:
        """
        Preprocess raw observation from robot into format expected by policy.
        
        Args:
            observation: Dict with keys:
                - 'observation.state': (7,) numpy array
                - 'observation.images.external': (H, W, 3) numpy array (RGB)
                - 'observation.images.wrist': (H, W, 3) numpy array (RGB)
        
        Returns:
            Preprocessed observation ready for policy
        """
        # Split state into arm_joints and gripper
        state = observation['observation.state']
        
        # Add a time dimension (T=1) to match training format: (T, D) for state and (T, H, W, C) for video
        processed = {
            'state.arm_joints': np.expand_dims(state[:6], axis=0),   # (1, 6)
            'state.gripper': np.expand_dims(state[6:7], axis=0),     # (1, 1)
            'video.external': np.expand_dims(observation['observation.images.external'], axis=0),  # (1, H, W, C)
            'video.wrist': np.expand_dims(observation['observation.images.wrist'], axis=0),        # (1, H, W, C)
        }
        
        return processed
    
    def get_action(self, observation: dict, language_instruction: str) -> np.ndarray:
        """
        Get action prediction from policy.
        
        Args:
            observation: Raw observation from robot
            language_instruction: Task description
        
        Returns:
            Action array of shape (action_horizon, 7) - normalized [0, 1]
        """
        # Preprocess observation
        processed_obs = self.preprocess_observation(observation)
        # Inject language instruction into processed observation under expected key
        # Use list so transforms handle it correctly
        processed_obs["annotation.human.task_description"] = [language_instruction]
        
        # Prepare observation for policy exactly like Gr00tPolicy.get_action does
        obs_copy = processed_obs.copy()
        # If not batched, add a batch dim (B=1) for all array-like fields
        try:
            # Use the policy helper to detect batching
            is_batch = self.policy._check_state_is_batched(obs_copy)
        except Exception:
            # Fallback: treat as not batched
            is_batch = False

        if not is_batch:
            # Unsqueeze values to add batch dim in the same way policy does
            from gr00t.model.policy import unsqueeze_dict_values

            obs_copy = unsqueeze_dict_values(obs_copy)

        # Ensure lists are converted to numpy arrays (e.g., language lists)
        for k, v in list(obs_copy.items()):
            if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
                try:
                    obs_copy[k] = np.array(v)
                except Exception:
                    obs_copy[k] = v

        # Apply policy transforms (policy expects batched inputs here)
        normalized_input = self.policy.apply_transforms(obs_copy)

        # Convert any PIL Image objects to numpy arrays (defensive)
        try:
            from PIL import Image as PILImageModule
            PILImageClass = getattr(PILImageModule, 'Image', PILImageModule)
            if not isinstance(PILImageClass, type) and hasattr(PILImageModule, 'Image'):
                PILImageClass = getattr(PILImageModule, 'Image')
        except Exception:
            PILImageClass = None

        def _convert_images(obj):
            if PILImageClass is not None and isinstance(obj, PILImageClass):
                try:
                    return np.asarray(obj)
                except Exception:
                    return obj
            if isinstance(obj, dict):
                return {k: _convert_images(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert_images(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_convert_images(v) for v in obj)
            return obj

        normalized_input = _convert_images(normalized_input)

        # Debug: optionally print a brief summary of normalized_input structure to locate bad types
        import os
        if os.environ.get("GR00T_DEBUG", "0") == "1":
            def _summarize(obj, path=""):
                lines = []
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        lines.extend(_summarize(v, path + "/" + str(k)))
                elif isinstance(obj, list) or isinstance(obj, tuple):
                    for i, v in enumerate(obj):
                        lines.extend(_summarize(v, path + f"/{i}"))
                else:
                    t = type(obj)
                    if isinstance(obj, np.ndarray):
                        lines.append((path, "ndarray", str(obj.dtype), obj.shape))
                    elif isinstance(obj, torch.Tensor):
                        lines.append((path, "tensor", str(obj.dtype), tuple(obj.shape)))
                    elif isinstance(obj, str):
                        lines.append((path, "str", str(len(obj))))
                    else:
                        lines.append((path, t.__name__))
                return lines

            summary = _summarize(normalized_input)
            print("[GR00T_DEBUG] normalized_input summary (first 80 entries):")
            for s in summary[:80]:
                print(" ", s)

        # Get normalized tensor from model
        normalized_tensor = self.policy._get_action_from_normalized_input(normalized_input)

        # Convert normalized tensor -> unnormalized action dict using policy's unapply transforms
        unnormalized_action = self.policy._get_unnormalized_action(normalized_tensor)

        # If we added a batch dim earlier, remove it using policy helper
        try:
            from gr00t.model.policy import squeeze_dict_values
        except Exception:
            squeeze_dict_values = None

        if not is_batch and squeeze_dict_values is not None:
            try:
                unnormalized_action = squeeze_dict_values(unnormalized_action)
            except Exception:
                pass

        # Extract action array
        action_arr = unnormalized_action.get('action') if isinstance(unnormalized_action, dict) else unnormalized_action

        # Convert torch.Tensor -> numpy if needed
        if isinstance(action_arr, torch.Tensor):
            action_np = action_arr.detach().cpu().numpy()
        else:
            action_np = np.array(action_arr)

        # If batch dim exists and B==1, squeeze it
        if action_np.ndim == 3 and action_np.shape[0] == 1:
            action_np = action_np[0]

        # At this point action_np should have shape (action_horizon, action_dim==7)
        return action_np
    
    def denormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """
        Convert normalized action [0, 1] to actual joint angles.
        
        Args:
            normalized_action: Normalized action (action_horizon, 7)
        
        Returns:
            Denormalized action in degrees
        """
        action_min = np.array(self.stats['action']['min'])
        action_max = np.array(self.stats['action']['max'])
        
        # Denormalize: x_real = x_norm * (max - min) + min
        denormalized = normalized_action * (action_max - action_min) + action_min
        
        return denormalized


def run_deployment(args):
    """Main deployment loop."""
    
    print("=" * 70)
    print("Kinova Gen3 GR00T Deployment")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Task: {args.task}")
    print(f"Max episodes: {args.max_episodes}")
    print(f"Device: {args.device}")
    print("=" * 70)
    
    # Initialize robot
    print("\n[1/3] Initializing robot...")
    robot = KinovaGen3Robot(
        ip_address=args.robot_ip,
        enable_cameras=True
    )
    
    # Initialize policy
    print("\n[2/3] Loading policy...")
    policy_deployer = KinovaPolicyDeployer(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Connect to robot
    print("\n[3/3] Connecting to robot...")
    robot.connect()
    
    try:
        # Run episodes
        for episode in range(args.max_episodes):
            print(f"\n{'='*70}")
            print(f"Episode {episode + 1}/{args.max_episodes}")
            print(f"Task: {args.task}")
            print(f"{'='*70}\n")
            
            # Reset to home
            robot.go_home()
            time.sleep(1.0)
            
            # Episode loop
            step = 0
            episode_start = time.time()
            
            while step < config.MAX_EPISODE_STEPS:
                step_start = time.time()
                
                # Get observation
                observation = robot.get_observation()
                
                # Get action from policy
                normalized_actions = policy_deployer.get_action(
                    observation=observation,
                    language_instruction=args.task
                )
                
                # Denormalize actions
                actions = policy_deployer.denormalize_action(normalized_actions)
                
                # Execute action chunk
                for i in range(config.ACTION_HORIZON):
                    action = actions[i]
                    
                    # Send to robot
                    robot.send_joint_positions(action, blocking=False)
                    
                    # Visualization
                    if args.visualize and step % 10 == 0:
                        # Display current observation
                        display_img = cv2.cvtColor(
                            observation['observation.images.external'],
                            cv2.COLOR_RGB2BGR
                        )
                        cv2.putText(
                            display_img,
                            f"Step: {step} | Action {i+1}/{config.ACTION_HORIZON}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )
                        cv2.imshow("Kinova Deployment", display_img)
                        cv2.waitKey(1)
                    
                    # Wait for control timestep
                    time.sleep(1.0 / config.CONTROL_FREQUENCY)
                    step += 1
                    
                    # Check for early termination
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nUser requested stop")
                        raise KeyboardInterrupt
                
                # Print progress
                elapsed = time.time() - step_start
                fps = config.ACTION_HORIZON / elapsed
                print(f"Step {step:3d} | FPS: {fps:.1f} | Action: {action[:3]}...", end='\r')
            
            episode_time = time.time() - episode_start
            print(f"\nEpisode {episode + 1} completed in {episode_time:.1f}s")
            
            # Return to home
            robot.go_home()
    
    except KeyboardInterrupt:
        print("\n\nDeployment interrupted by user")
    
    except Exception as e:
        print(f"\n\nError during deployment: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\nCleaning up...")
        robot.disconnect()
        cv2.destroyAllWindows()
        print("Deployment ended safely")


def main():
    parser = argparse.ArgumentParser(description="Deploy GR00T on Kinova Gen3")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(config.DEFAULT_CHECKPOINT),
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=config.DEFAULT_TASK,
        help="Language instruction for the task"
    )
    parser.add_argument(
        "--robot-ip",
        type=str,
        default=config.ROBOT_IP,
        help="Robot IP address"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.DEVICE,
        help="Device for inference (cuda:0 or cpu)"
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=1,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=config.ENABLE_VISUALIZATION,
        help="Show camera feed during deployment"
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    run_deployment(args)


if __name__ == "__main__":
    main()
