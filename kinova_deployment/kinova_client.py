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
Kinova Robot Client - Controls Kinova arm using policy server

This script:
1. Connects to the Kinova robot and cameras
2. Connects to the policy server (running on GPU machine)
3. Captures observations (images + joint states)
4. Sends observations to server → receives actions
5. Executes actions on the robot

Usage:
    # Default (assumes server on localhost:5555)
    python kinova_client.py
    
    # Connect to remote server
    python kinova_client.py --host 192.168.1.100 --port 5555 --task "pick up the object"
"""

# Add parent directory to path for imports
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

import cv2
import numpy as np
from tqdm import tqdm

from kinova_deployment.kinova_robot import KinovaGen3Robot
import kinova_deployment.config as config

import matplotlib.pyplot as plt

# Setup figure once (before loop)
plt.ion()  # Interactive mode for continuous updates
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.set_title("Random frame")
# # Create and setup OpenCV window
# window_name = "Camera View"
# cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# cv2.resizeWindow(window_name, 1280, 480)

from gr00t.eval.robot import RobotInferenceClient # Import delays cause CV2 to crash / segfault.


class KinovaPolicyClient:
    """Client that connects Kinova robot to GR00T policy server."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        language_instruction: str = "Pick and place the object",
    ):
        """
        Initialize Kinova policy client.
        
        Args:
            host: Policy server host address
            port: Policy server port
            language_instruction: Task description for the policy
        """
        self.host = host
        self.port = port
        self.language_instruction = language_instruction
        
        print(f"Connecting to policy server at {host}:{port}...")
        self.policy_client = RobotInferenceClient(host=host, port=port)
        
        # Test connection
        try:
            modality_config = self.policy_client.get_modality_config()
            print(f"✓ Connected to policy server")
            print(f"  Server modality config: {list(modality_config.keys())}")
        except Exception as e:
            print(f"✗ Failed to connect to policy server: {e}")
            print(f"  Make sure the server is running:")
            print(f"  python kinova_deployment/kinova_server.py --checkpoint <path>")
            raise
    
    def get_action(self, external_img: np.ndarray, wrist_img: np.ndarray, state: np.ndarray):
        """
        Get action from policy server.
        
        Args:
            external_img: External camera image (H, W, 3) RGB
            wrist_img: Wrist camera image (H, W, 3) RGB
            state: Current joint states (7,) [6 joints + gripper]
            
        Returns:
            dict with action.arm_joints and action.gripper
        """
        # Prepare observation dict matching server's expected format
        obs_dict = {
            "video.external": np.expand_dims(external_img, axis=0),  # Add time dimension
            "video.wrist": np.expand_dims(wrist_img, axis=0),
            "state.arm_joints": np.expand_dims(state[:6], axis=0),
            "state.gripper": np.expand_dims(state[6:7], axis=0),
            "annotation.human.task_description": [self.language_instruction],
        }
        
        # Get action from server
        action = self.policy_client.get_action(obs_dict)
        return action


def main():
    parser = argparse.ArgumentParser(description="Kinova Robot Client")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Policy server host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="Policy server port (default: 5555)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Pick and place the object",
        help="Task description for the policy"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Number of control steps to execute (default: 100)"
    )
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=16,
        help="Action horizon (default: 16)"
    )
    parser.add_argument(
        "--actions_per_step",
        type=int,
        default=16,
        help="Number of actions to execute per step (default: 4)"
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save video of execution"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Require confirmation before each action step (very safe mode)"
    )
    parser.add_argument(
        "--delay_per_action",
        type=float,
        default=0.05,
        help="Delay between each action in seconds (default: 0.05 = 20Hz). Increase for slower motion (e.g., 0.2 = 5Hz)"
    )
    parser.add_argument(
        "--scale_actions",
        type=float,
        default=1.0,
        help="Scale factor for actions (0.0-1.0). Use 0.1-0.3 for very conservative movement (default: 1.0)"
    )
    parser.add_argument(
        "--max_joint_delta",
        type=float,
        default=None,
        help="Maximum joint angle change per action in degrees (default: None). Use 1.0-5.0 for safety"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run mode - get actions but don't send to robot (for testing)"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("Kinova GR00T Robot Client")
    print("=" * 70)
    print(f"Server: {args.host}:{args.port}")
    print(f"Task: {args.task}")
    print(f"Steps: {args.steps}")
    print(f"Action horizon: {args.action_horizon}")
    print(f"Actions per step: {args.actions_per_step}")
    print(f"Delay per action: {args.delay_per_action}s ({1.0/args.delay_per_action:.1f}Hz)")
    
    # Safety settings
    if args.scale_actions != 1.0:
        print(f"⚠ Action scaling: {args.scale_actions*100:.0f}% (conservative mode)")
    if args.max_joint_delta:
        print(f"⚠ Max joint delta: {args.max_joint_delta}° per action")
    if args.interactive:
        print(f"⚠ Interactive mode: Will prompt before each step")
    if args.dry_run:
        print(f"⚠ DRY RUN MODE: Actions will NOT be sent to robot")
    
    print("=" * 70)
    
    # Initialize policy client
    policy_client = KinovaPolicyClient(
        host=args.host,
        port=args.port,
        language_instruction=args.task,
    )
    
    # Initialize robot
    robot = KinovaGen3Robot(enable_cameras=True)
    
    # Video writer setup
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            f'kinova_execution_{int(time.time())}.mp4',
            fourcc,
            30.0,
            (1280, 480)  # Combined camera view
        )
    
    try:
        with robot.activate():
            print(f"\n▶ Starting execution...")
            print(f"  Task: {args.task}")
            if args.dry_run:
                print(f"  DRY RUN - Not sending commands to robot")
            print(f"  Press Ctrl+C to stop\n")
            
            # Wait for user confirmation to start
            if args.interactive or args.dry_run:
                input("Press ENTER to start execution (or Ctrl+C to cancel)...")
            
            last_joint_positions = robot.get_joint_states()
            
            for step in tqdm(range(args.steps), desc="Executing policy"):
                # Get current observation
                images = robot.get_camera_images()
                state = robot.get_joint_states()
                
                external_frame = images["external"]
                wrist_frame = images["wrist"]
                # Resize frames to same height for side-by-side display
                h = min(external_frame.shape[0], wrist_frame.shape[0])
                external_resized = cv2.resize(external_frame, 
                                                (int(external_frame.shape[1] * h / external_frame.shape[0]), h))
                wrist_resized = cv2.resize(wrist_frame, 
                                            (int(wrist_frame.shape[1] * h / wrist_frame.shape[0]), h))
                
                # Add labels
                cv2.putText(external_resized, "External View", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(wrist_resized, "Wrist View", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2display = np.hstack([external_resized, wrist_resized])
                # cv2.imshow(window_name, cv2display)
                # cv2.waitKey(1)  # 1m`s delay, non-blocking
                # Update display
                ax.clear()
                ax.imshow(cv2display)
                ax.set_title("Random frame")
                plt.pause(0.01)  # Refresh every 10ms (~100 FPS)
                plt.draw()

                # Get action from policy server
                start_time = time.time()
                action = policy_client.get_action(
                    images['external'],
                    images['wrist'],
                    state
                )
                inference_time = time.time() - start_time
                
                # Interactive mode: show action and ask for confirmation
                if args.interactive and not args.dry_run:
                    print(f"\n[Step {step}] Action preview:")
                    print(f"  Inference time: {inference_time*1000:.1f}ms")
                    modality_keys = ["arm_joints", "gripper"]
                    first_action = np.concatenate(
                        [np.atleast_1d(action[f"action.{key}"][0]) for key in modality_keys],
                        axis=0,
                    )
                    delta = first_action - state
                    print(f"  Joint deltas (degrees): {delta}")
                    print(f"  Max delta: {np.abs(delta).max():.2f}°")
                    response = input("  Execute this action? [y/N/q]: ").lower()
                    if response == 'q':
                        print("Quitting...")
                        break
                    elif response != 'y':
                        print("Skipping this action")
                        continue
                
                # Execute actions
                modality_keys = ["arm_joints", "gripper"]
                actions_executed = 0
                
                for i in range(min(args.actions_per_step, args.action_horizon)):
                    # Concatenate arm joints + gripper
                    concat_action = np.concatenate(
                        [np.atleast_1d(action[f"action.{key}"][i]) for key in modality_keys],
                        axis=0,
                    )
                    
                    assert concat_action.shape == (7,), f"Expected (7,) got {concat_action.shape}"
                    
                    # Get current position for delta calculations
                    current_pos = robot.get_joint_states()
                    
                    # Apply action scaling (for conservative movement)
                    if args.scale_actions != 1.0:
                        # Scale the delta from current position
                        action_delta = concat_action - current_pos
                        concat_action = current_pos + (action_delta * args.scale_actions)
                    
                    # Apply max joint delta limit
                    if args.max_joint_delta is not None:
                        action_delta = concat_action - current_pos
                        # Clip each joint delta
                        action_delta = np.clip(action_delta, -args.max_joint_delta, args.max_joint_delta)
                        concat_action = current_pos + action_delta
                    
                    # Check if action would violate joint limits
                    joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper']
                    action_valid = True
                    
                    for j, (pos, name) in enumerate(zip(concat_action, joint_names)):
                        min_val, max_val = config.JOINT_LIMITS[name]
                        safe_min = min_val + config.SAFETY_MARGIN
                        safe_max = max_val - config.SAFETY_MARGIN
                        
                        # if not (safe_min <= pos <= safe_max):
                        #     if step % 10 == 0 or i == 0:  # Only print occasionally to avoid spam
                        #         print(f"  ⚠ {name} would exceed limits: {pos:.1f}° (range: {safe_min:.1f}-{safe_max:.1f}°)")
                        #     # Clip to safe range
                        #     # if name == "joint_4":
                        #     #     # Wrap-around joint
                        #     #     while pos < 0:
                        #     #         pos += 360.0
                        #     #     while pos > 360.0:
                        #     #         pos -= 360.0
                        #     concat_action[j] = np.clip(pos, safe_min, safe_max)
                    
                    # Send to robot (unless dry run)
                    if not args.dry_run:
                        try:
                            robot.send_joint_positions(concat_action, blocking=False)
                            actions_executed += 1
                            last_joint_positions = concat_action
                        except Exception as e:
                            print(f"  ✗ Failed to send action {i}: {e}")
                            break  # Stop this action sequence if send fails
                    else:
                        # In dry run, just print what we would send
                        if i == 0 and step % 10 == 0:
                            delta = concat_action - state
                            print(f"  [Dry run] Would send: {concat_action}, delta: {delta}")
                        actions_executed += 1
                    
                    # Save video frame
                    if video_writer is not None:
                        images = robot.get_camera_images()
                        combined = np.hstack([
                            cv2.cvtColor(images['external'], cv2.COLOR_RGB2BGR),
                            cv2.cvtColor(images['wrist'], cv2.COLOR_RGB2BGR)
                        ])
                        video_writer.write(combined)
                    
                    time.sleep(args.delay_per_action)
                
                if step % 10 == 0:
                    print(f"  Step {step}: Inference: {inference_time*1000:.1f}ms, Executed: {actions_executed}/{min(args.actions_per_step, args.action_horizon)} actions")
            
            print(f"\n✓ Execution complete!")
            
            # Return to home
            if not args.dry_run:
                print("Returning to home position...")
                robot.go_home()
            
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        print("Returning to home position...")
        robot.go_home()
        
    except Exception as e:
        print(f"\n✗ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"✓ Video saved")
        
        print("Disconnecting from robot...")


if __name__ == "__main__":
    main()
