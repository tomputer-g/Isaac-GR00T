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
SAFE TEST: Check cameras only (NO robot connection).

This script tests camera access and image capture.
NO robot connection, NO physical movement.
"""

import time
import cv2
import numpy as np

import config


def test_cameras():
    """Test USB camera access and image capture."""
    print("=" * 80)
    print("SAFE TEST #2: Camera Access Only (NO ROBOT)")
    print("=" * 80)
    print("\nThis test will:")
    print("  ✓ Access USB cameras")
    print("  ✓ Capture test images")
    print("  ✓ Display live feed for 10 seconds")
    print("  ✗ NOT connect to robot")
    print("  ✗ NOT move anything")
    print("\n" + "=" * 80 + "\n")
    
    external_camera = None
    wrist_camera = None
    
    try:
        # Test external camera
        print(f"[1/4] Testing external camera (index {config.EXTERNAL_CAMERA_INDEX})...")
        external_camera = cv2.VideoCapture(config.EXTERNAL_CAMERA_INDEX)
        
        if not external_camera.isOpened():
            print(f"❌ ERROR: Cannot open external camera at index {config.EXTERNAL_CAMERA_INDEX}")
            print("\nTroubleshooting:")
            print("  - Check USB connection")
            print("  - Try different index in config.py (0, 1, 2, etc.)")
            print("  - List available cameras:")
            print("    Linux: ls /dev/video*")
            print("    macOS: system_profiler SPCameraDataType")
            return False
        
        external_camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        external_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        external_camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        
        ret, frame = external_camera.read()
        if not ret:
            print(f"❌ ERROR: Cannot read from external camera")
            return False
        
        print(f"✓ External camera working")
        print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
        print(f"  Channels: {frame.shape[2]}")
        
        # Test wrist camera
        print(f"\n[2/4] Testing wrist camera (index {config.WRIST_CAMERA_INDEX})...")
        wrist_camera = cv2.VideoCapture(config.WRIST_CAMERA_INDEX)
        
        if not wrist_camera.isOpened():
            print(f"❌ ERROR: Cannot open wrist camera at index {config.WRIST_CAMERA_INDEX}")
            print("\nNote: If you only have one camera, update config.py:")
            print("  WRIST_CAMERA_INDEX = EXTERNAL_CAMERA_INDEX")
            return False
        
        wrist_camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        wrist_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        wrist_camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        
        ret, frame = wrist_camera.read()
        if not ret:
            print(f"❌ ERROR: Cannot read from wrist camera")
            return False
        
        print(f"✓ Wrist camera working")
        print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
        print(f"  Channels: {frame.shape[2]}")
        
        # Capture test images
        print(f"\n[3/4] Capturing test images...")
        ret1, ext_img = external_camera.read()
        ret2, wrist_img = wrist_camera.read()
        
        if ret1 and ret2:
            print(f"✓ Both cameras captured successfully")
            
            # Save test images
            cv2.imwrite("test_external_camera.jpg", ext_img)
            cv2.imwrite("test_wrist_camera.jpg", wrist_img)
            print(f"✓ Test images saved:")
            print(f"  - test_external_camera.jpg")
            print(f"  - test_wrist_camera.jpg")
        
        # Display live feed
        print(f"\n[4/4] Displaying live feed for 10 seconds...")
        print(f"  Press 'q' to quit early")
        
        start_time = time.time()
        frame_count = 0
        
        while (time.time() - start_time) < 10.0:
            ret1, ext_img = external_camera.read()
            ret2, wrist_img = wrist_camera.read()
            
            if not (ret1 and ret2):
                print("Warning: Frame capture failed")
                continue
            
            # Combine images side by side
            combined = np.hstack([ext_img, wrist_img])
            
            # Add text overlays
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
            
            elapsed = time.time() - start_time
            cv2.putText(
                combined,
                f"Time: {elapsed:.1f}s / 10.0s",
                (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            cv2.imshow("Camera Test - Press 'q' to quit", combined)
            
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        fps = frame_count / (time.time() - start_time)
        print(f"\n✓ Live feed test complete")
        print(f"  Captured {frame_count} frames")
        print(f"  Average FPS: {fps:.1f}")
        
        cv2.destroyAllWindows()
        
        # Summary
        print("\n" + "=" * 80)
        print("TEST RESULTS:")
        print("=" * 80)
        print("✓ External camera accessible")
        print("✓ Wrist camera accessible")
        print("✓ Image capture working")
        print("✓ Live feed working")
        print(f"✓ FPS: {fps:.1f} (target: {config.CAMERA_FPS})")
        
        print("\n" + "=" * 80)
        print("NEXT STEPS:")
        print("=" * 80)
        print("Cameras are working! You can now:")
        print("  1. Test robot connection: python test_robot_connection.py")
        print("  2. Deploy on real robot: python deploy_kinova.py")
        print("=" * 80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        if external_camera is not None:
            external_camera.release()
        if wrist_camera is not None:
            wrist_camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    success = test_cameras()
    sys.exit(0 if success else 1)
