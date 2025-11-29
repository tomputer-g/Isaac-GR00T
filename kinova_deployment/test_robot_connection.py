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

"""Test robot connection and basic functionality."""

import time
import cv2
import numpy as np

import config
from kinova_robot import KinovaGen3Robot
from termcolor import colored

# print(colored("This text is green!", 'green'))
def test_connection():
    """Test basic robot connection."""
    print("=" * 70)
    print("Kinova Gen3 Connection Test")
    print("=" * 70)
    print(f"Robot IP: {config.ROBOT_IP}")
    print(f"Robot Port: {config.ROBOT_PORT}")
    print("=" * 70)
    
    robot = KinovaGen3Robot()
    
    try:
        # Test connection
        print("\n[1/5] Testing robot connection...")
        robot.connect()
        print("✓ Robot connected successfully")
        
        # Test reading joint states
        print("\n[2/5] Reading joint states...")
        state = robot.get_joint_states()
        print(f"✓ Current joint states (degrees):")
        for i, pos in enumerate(state[:6]):
            print(f"  Joint {i+1}: {pos:.2f}°")
        print(f"  Gripper: {state[6]:.2f}")
        
        # Test cameras
        print("\n[3/5] Testing cameras...")
        images = robot.get_camera_images()
        print(f"✓ External camera: {images['external'].shape}")
        print(f"✓ Wrist camera: {images['wrist'].shape}")
        
        # Display camera feeds for 5 seconds
        print("\n[4/5] Testing camera capture (5 seconds)...")
        print("  Note: Saving frames to disk instead of displaying to avoid OpenCV/RealSense conflicts")
        
        start_time = time.time()
        frame_count = 0
        save_interval = 1.0  # Save every second
        last_save_time = start_time
        
        while (time.time() - start_time) < 5.0:
            try:
                images = robot.get_camera_images()
                
                # Validate images
                if images['external'] is None or images['wrist'] is None:
                    print("⚠ Warning: Got None image, skipping frame")
                    time.sleep(0.033)  # ~30fps
                    continue
                
                frame_count += 1
                
                # Save a frame every second
                current_time = time.time()
                if current_time - last_save_time >= save_interval:
                    # Combine images side by side
                    combined = np.hstack([
                        cv2.cvtColor(images['external'], cv2.COLOR_RGB2BGR),
                        cv2.cvtColor(images['wrist'], cv2.COLOR_RGB2BGR)
                    ])
                    
                    cv2.putText(
                        combined,
                        "External Camera",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        combined,
                        "Wrist Camera",
                        (650, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
                    filename = f"test_robot_cameras_{int(current_time - start_time)}s.jpg"
                    cv2.imwrite(filename, combined)
                    print(f"  Saved: {filename}")
                    last_save_time = current_time
                
                time.sleep(0.033)  # ~30fps
                    
            except Exception as display_err:
                print(f"⚠ Frame capture error: {display_err}")
                time.sleep(0.033)
                continue
        
        print(f"✓ Camera test complete ({frame_count} frames captured)")
        print("  Check test_robot_cameras_*.jpg for saved frames")
        
        # Test observation collection
        print("\n[5/5] Testing observation collection...")
        obs = robot.get_observation()
        print(f"✓ Observation keys: {list(obs.keys())}")
        print(f"  State shape: {obs['observation.state'].shape}")
        print(f"  External image shape: {obs['observation.images.external'].shape}")
        print(f"  Wrist image shape: {obs['observation.images.wrist'].shape}")
        
        print("\n" + "=" * 70)
        print("All tests passed! ✓")
        print("=" * 70)
        print("\nYou can now run deployment with:")
        print("  python deploy_kinova.py --checkpoint <path> --task <task>")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        robot.disconnect()


if __name__ == "__main__":
    test_connection()
