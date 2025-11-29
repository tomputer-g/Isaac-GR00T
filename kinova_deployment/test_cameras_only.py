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
import os
import glob
import pwd
import grp
import getpass

import config


# Optional RealSense support
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except Exception:
    rs = None
    REALSENSE_AVAILABLE = False

class RealSenseCapture:
    """Simple wrapper to provide a VideoCapture-like interface for Intel RealSense color stream.

    Usage: cap = RealSenseCapture(device_index=0, width=640, height=480, fps=30)
    Methods: isOpened(), read() -> (ret, frame), release(), set(prop, value)
    """
    def __init__(self, device_index=0, width=None, height=None, fps=None):
        if not REALSENSE_AVAILABLE:
            self._ok = False
            return
        try:
            self._ctx = rs.context()
            devices = self._ctx.query_devices()
            if len(devices) == 0:
                self._ok = False
                return
            # choose device by index
            if device_index < len(devices):
                dev = devices[device_index]
            else:
                dev = devices[0]
            self._serial = dev.get_info(rs.camera_info.serial_number)

            self._pipeline = rs.pipeline()
            self._config = rs.config()
            self._config.enable_device(self._serial)

            # store properties so set() can update them
            self._width = int(width) if width is not None else int(config.CAMERA_WIDTH)
            self._height = int(height) if height is not None else int(config.CAMERA_HEIGHT)
            self._fps = int(fps) if fps is not None else int(config.CAMERA_FPS)

            self._config.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)
            self._pipeline.start(self._config)
            self._ok = True
        except Exception as e:
            print(f"RealSense init failed: {e}")
            self._ok = False

    def isOpened(self):
        return getattr(self, "_ok", False)

    def set(self, prop, value):
        """Mimic cv2.VideoCapture.set(). Accept width/height/fps. Does not restart pipeline.

        Returns True if the property is accepted (but may not take effect until next init).
        """
        try:
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                self._width = int(value)
                return True
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                self._height = int(value)
                return True
            if prop == cv2.CAP_PROP_FPS:
                self._fps = int(value)
                return True
        except Exception:
            pass
        return False

    def read(self):
        if not self.isOpened():
            return False, None
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=5000)
            color = frames.get_color_frame()
            if not color:
                return False, None
            # convert to numpy
            import numpy as _np
            img = _np.asanyarray(color.get_data())
            return True, img
        except Exception as e:
            print(f"RealSense read failed: {e}")
            return False, None

    def release(self):
        try:
            if getattr(self, "_pipeline", None) is not None:
                self._pipeline.stop()
        except Exception:
            pass


def try_open_camera(index):
    """Try to open a camera using several candidate names and OpenCV backends.

    Returns (cv2.VideoCapture, candidate_used, api_used) or (None, None, None) on failure.
    """
    # Start with configured index and device path
    candidates = [index, f"/dev/video{index}"]

    # Add all available /dev/video* nodes as additional candidates (sorted)
    try:
        devs = sorted(glob.glob('/dev/video*'))
        for d in devs:
            if d not in candidates:
                candidates.append(d)
    except Exception:
        pass

    # build a list of OpenCV API preferences we can try
    backends = []
    for api_name in ("CAP_V4L2", "CAP_GSTREAMER", "CAP_FFMPEG", "CAP_ANY"):
        api = getattr(cv2, api_name, None)
        if api is not None:
            backends.append(api)

    # Try each candidate with each backend
    for candidate in candidates:
        # Normalize candidate: if it's numeric-like, pass integer to VideoCapture
        try:
            cand_int = int(candidate)
        except Exception:
            cand_int = None

        for api in backends:
            try:
                if cand_int is not None:
                    cap = cv2.VideoCapture(cand_int, api)
                else:
                    cap = cv2.VideoCapture(candidate, api)
            except Exception:
                try:
                    if cand_int is not None:
                        cap = cv2.VideoCapture(cand_int)
                    else:
                        cap = cv2.VideoCapture(candidate)
                except Exception:
                    cap = None

            if cap is not None and cap.isOpened():
                return cap, candidate, api

        # Also try without specifying backend (some backends misreport availability)
        try:
            if cand_int is not None:
                cap = cv2.VideoCapture(cand_int)
            else:
                cap = cv2.VideoCapture(candidate)
        except Exception:
            cap = None

        if cap is not None and cap.isOpened():
            return cap, candidate, None

    # Final fallback: try default open with index
    try:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap, index, None
    except Exception:
        pass

    # If OpenCV failed, try RealSense via pyrealsense2 (if available)
    if REALSENSE_AVAILABLE:
        try:
            rs_cap = RealSenseCapture(device_index=index)
            if rs_cap.isOpened():
                return rs_cap, f"realsense[{index}]", "pyrealsense2"
        except Exception:
            pass

    return None, None, None


def print_device_diagnostics():
    """Print helpful diagnostics about /dev/video* devices and current user/group."""
    devices = glob.glob('/dev/video*')
    if not devices:
        print("No /dev/video* nodes found.")
    else:
        print("Available video devices and permissions:")
        for d in devices:
            try:
                st = os.stat(d)
                owner = pwd.getpwuid(st.st_uid).pw_name
                group = grp.getgrgid(st.st_gid).gr_name
                perms = oct(st.st_mode & 0o777)
                print(f"  {d}: owner={owner}, group={group}, perms={perms}")
            except Exception as e:
                print(f"  {d}: stat error: {e}")

    user = getpass.getuser()
    try:
        primary = grp.getgrgid(os.getgid()).gr_name
    except Exception:
        primary = None
    # groups where the user is a member (supplementary groups)
    supplementary = [g.gr_name for g in grp.getgrall() if user in g.gr_mem]
    groups = []
    if primary:
        groups.append(primary)
    groups.extend(supplementary)
    print(f"Current user: {user}")
    print(f"Groups: {groups}")


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
        # If RealSense SDK is available and a device is present, try it first
        external_camera = None
        used_ext_candidate = None
        used_ext_api = None
        try:
            if REALSENSE_AVAILABLE:
                ctx = rs.context()
                devices = ctx.query_devices()
                if len(devices) > 0:
                    print("Attempting to open RealSense color stream via pyrealsense2...")
                    rs_cap = RealSenseCapture(device_index=0)
                    if rs_cap.isOpened():
                        external_camera = rs_cap
                        used_ext_candidate = "realsense[0]"
                        used_ext_api = "pyrealsense2"
        except Exception:
            pass

        if external_camera is None:
            external_camera, used_ext_candidate, used_ext_api = try_open_camera(config.EXTERNAL_CAMERA_INDEX)
        
        if external_camera is None:
            return False
        
        else:
            print(f"✓ Opened external camera (candidate={used_ext_candidate}, api={used_ext_api})")

            external_camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            external_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            external_camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
            
            ret, frame = external_camera.read()
            if not ret:
                print(f"❌ ERROR: Cannot read from external camera")
                return False

        # Test wrist camera
        print(f"\n[2/4] Testing wrist camera...")
        
        # If RealSense SDK is available and a device is present, try it first
        wrist_camera = None
        used_wrist_candidate = None
        used_wrist_api = None
        try:
            if REALSENSE_AVAILABLE:
                ctx = rs.context()
                devices = ctx.query_devices()
                if len(devices) > 0:
                    print("Attempting to open RealSense color stream via pyrealsense2...")
                    rs_cap = RealSenseCapture(device_index=1)
                    if rs_cap.isOpened():
                        wrist_camera = rs_cap
                        used_wrist_candidate = "realsense[1]"
                        used_wrist_api = "pyrealsense2"
        except Exception:
            pass

        if wrist_camera is None:
            wrist_camera, used_wrist_candidate, used_wrist_api = try_open_camera(config.WRIST_CAMERA_INDEX)

        if wrist_camera is None:
            return False
        
        else:
            print(f"✓ Opened wrist camera (candidate={used_wrist_candidate}, api={used_wrist_api})")

            wrist_camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            wrist_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            wrist_camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

            ret, frame = wrist_camera.read()
            if not ret:
                print(f"❌ ERROR: Cannot read from wrist camera")
                return False
        
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
