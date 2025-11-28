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

"""Kinova Gen3 robot control interface."""

import time
import numpy as np
import cv2
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

try:
    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2
    from kortex_api.TCPTransport import TCPTransport
    from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
    from kortex_api.SessionManager import SessionManager
    KORTEX_AVAILABLE = True
except ImportError:
    KORTEX_AVAILABLE = False
    print("WARNING: kortex_api not installed. Robot control will not work.")
    print("Install with: pip install kortex_api")

import config


class KinovaGen3Robot:
    """
    Interface to Kinova Gen3 7-DOF robotic arm.
    
    Provides methods to:
    - Connect/disconnect from robot
    - Read current joint states
    - Send joint position commands
    - Control gripper
    - Capture camera images
    """
    
    def __init__(
        self,
        ip_address: str = config.ROBOT_IP,
        port: int = config.ROBOT_PORT,
        username: str = config.ROBOT_USERNAME,
        password: str = config.ROBOT_PASSWORD,
        enable_cameras: bool = True,
    ):
        """Initialize Kinova robot interface."""
        if not KORTEX_AVAILABLE:
            raise ImportError("kortex_api is required for robot control")
        
        self.ip_address = ip_address
        self.port = port
        self.username = username
        self.password = password
        self.enable_cameras = enable_cameras
        
        # Robot connection objects
        self.router: Optional[RouterClient] = None
        self.base: Optional[BaseClient] = None
        self.base_cyclic: Optional[BaseCyclicClient] = None
        self.is_connected = False
        
        # Camera objects
        self.external_camera = None
        self.wrist_camera = None
        
        print(f"Kinova Gen3 Robot initialized (IP: {ip_address})")
    
    def connect(self):
        """Connect to the Kinova robot and initialize cameras."""
        if self.is_connected:
            print("Already connected to robot")
            return
        
        try:
            print(f"Connecting to robot at {self.ip_address}:{self.port}...")
            
            # Create connection
            self.router = RouterClient(TCPTransport(), RouterClient.ROUTER_ADDRESS, self.port)
            
            # Create session
            session_info = SessionManager.CreateRouterSession(
                self.username,
                self.password,
                self.ip_address,
                self.port
            )
            
            # Create base client for high-level control
            self.base = BaseClient(self.router)
            
            # Create base cyclic client for real-time feedback
            self.base_cyclic = BaseCyclicClient(self.router)
            
            self.is_connected = True
            print("Connected to Kinova Gen3")
            
            # Initialize cameras
            if self.enable_cameras:
                self._setup_cameras()
            
            # Move to home position
            print("Moving to home position...")
            self.go_home()
            
        except Exception as e:
            print(f"Failed to connect to robot: {e}")
            self.is_connected = False
            raise
    
    def disconnect(self):
        """Disconnect from robot and release cameras."""
        if not self.is_connected:
            return
        
        print("Disconnecting from robot...")
        
        # Release cameras
        if self.external_camera is not None:
            self.external_camera.release()
            self.external_camera = None
        
        if self.wrist_camera is not None:
            self.wrist_camera.release()
            self.wrist_camera = None
        
        # Disconnect from robot
        if self.router is not None:
            self.router.disconnect()
            self.router = None
        
        self.base = None
        self.base_cyclic = None
        self.is_connected = False
        print("Disconnected from robot")
    
    @contextmanager
    def activate(self):
        """Context manager for robot connection."""
        try:
            self.connect()
            yield self
        finally:
            self.disconnect()
    
    def _setup_cameras(self):
        """Initialize USB cameras."""
        try:
            print("Setting up cameras...")
            
            # External camera
            self.external_camera = cv2.VideoCapture(config.EXTERNAL_CAMERA_INDEX)
            self.external_camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.external_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            self.external_camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
            
            # Wrist camera
            self.wrist_camera = cv2.VideoCapture(config.WRIST_CAMERA_INDEX)
            self.wrist_camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.wrist_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            self.wrist_camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
            
            print("Cameras initialized")
        except Exception as e:
            print(f"Failed to setup cameras: {e}")
            self.external_camera = None
            self.wrist_camera = None
    
    def get_joint_states(self) -> np.ndarray:
        """
        Get current joint positions.
        
        Returns:
            np.ndarray: Array of shape (7,) with [6 joint angles in degrees, gripper position 0-100]
        """
        if not self.is_connected:
            raise RuntimeError("Robot not connected")
        
        try:
            # Get joint angles using BaseCyclic for faster feedback
            feedback = self.base_cyclic.RefreshFeedback()
            
            # Extract joint angles (in degrees)
            joint_angles = []
            for actuator in feedback.actuators[:6]:  # First 6 are arm joints
                joint_angles.append(actuator.position)
            
            # Get gripper position (convert from 0-1 to 0-100)
            gripper_feedback = self.base.GetMeasuredGripperMovement()
            gripper_position = gripper_feedback.finger[0].value * 100.0
            
            state = np.array(joint_angles + [gripper_position], dtype=np.float32)
            return state
            
        except Exception as e:
            print(f"Error reading joint states: {e}")
            # Return zeros as fallback
            return np.zeros(7, dtype=np.float32)
    
    def send_joint_positions(self, target_positions: np.ndarray, blocking: bool = False):
        """
        Send target joint positions to robot.
        
        Args:
            target_positions: Array of shape (7,) with [6 joint angles in degrees, gripper 0-100]
            blocking: If True, wait for motion to complete
        """
        if not self.is_connected:
            raise RuntimeError("Robot not connected")
        
        if target_positions.shape != (7,):
            raise ValueError(f"Expected shape (7,), got {target_positions.shape}")
        
        # Validate positions are within limits
        if not self._is_safe_position(target_positions):
            print("WARNING: Target position outside safe limits, clipping...")
            target_positions = self._clip_to_safe_limits(target_positions)
        
        try:
            # Send arm joint positions
            action = Base_pb2.Action()
            action.name = "goto_joint_positions"
            action.application_data = ""
            
            # Create reach joint angles action
            reach_joint_angles = action.reach_joint_angles
            reach_joint_angles.joint_angles.joint_angles.clear()
            
            for i in range(6):
                joint_angle = reach_joint_angles.joint_angles.joint_angles.add()
                joint_angle.value = float(target_positions[i])
            
            # Send command
            self.base.ExecuteAction(action)
            
            # Send gripper command
            self._set_gripper_position(target_positions[6])
            
            # Wait if blocking
            if blocking:
                self._wait_for_motion_complete()
                
        except Exception as e:
            print(f"Error sending joint positions: {e}")
    
    def _set_gripper_position(self, position: float):
        """Set gripper position (0-100)."""
        try:
            gripper_command = Base_pb2.GripperCommand()
            gripper_command.mode = Base_pb2.GRIPPER_POSITION
            
            finger = gripper_command.gripper.finger.add()
            finger.finger_identifier = 1
            finger.value = position / 100.0  # Convert to 0-1 range
            
            self.base.SendGripperCommand(gripper_command)
        except Exception as e:
            print(f"Error setting gripper: {e}")
    
    def _wait_for_motion_complete(self, timeout: float = 5.0):
        """Wait for robot motion to complete."""
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            # Check if robot is still moving
            feedback = self.base_cyclic.RefreshFeedback()
            
            # Simple check: if all velocities are near zero, motion is complete
            velocities = [abs(actuator.velocity) for actuator in feedback.actuators[:6]]
            if all(v < 1.0 for v in velocities):  # Threshold: 1 deg/sec
                return
            
            time.sleep(0.1)
        
        print("WARNING: Motion timeout reached")
    
    def _is_safe_position(self, positions: np.ndarray) -> bool:
        """Check if positions are within safe joint limits."""
        joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper']
        
        for i, (pos, name) in enumerate(zip(positions, joint_names)):
            min_val, max_val = config.JOINT_LIMITS[name]
            # Add safety margin
            safe_min = min_val + config.SAFETY_MARGIN
            safe_max = max_val - config.SAFETY_MARGIN
            
            if not (safe_min <= pos <= safe_max):
                print(f"Joint {name} out of range: {pos:.2f} (safe range: {safe_min:.2f}-{safe_max:.2f})")
                return False
        
        return True
    
    def _clip_to_safe_limits(self, positions: np.ndarray) -> np.ndarray:
        """Clip positions to safe joint limits."""
        clipped = positions.copy()
        joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper']
        
        for i, name in enumerate(joint_names):
            min_val, max_val = config.JOINT_LIMITS[name]
            safe_min = min_val + config.SAFETY_MARGIN
            safe_max = max_val - config.SAFETY_MARGIN
            clipped[i] = np.clip(positions[i], safe_min, safe_max)
        
        return clipped
    
    def get_camera_images(self) -> Dict[str, np.ndarray]:
        """
        Capture images from both cameras.
        
        Returns:
            Dict with keys 'external' and 'wrist', values are (H, W, 3) RGB arrays
        """
        images = {}
        
        if self.external_camera is not None:
            ret, frame = self.external_camera.read()
            if ret:
                # Convert BGR to RGB
                images['external'] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                print("Failed to read external camera")
                images['external'] = np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
        else:
            images['external'] = np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
        
        if self.wrist_camera is not None:
            ret, frame = self.wrist_camera.read()
            if ret:
                images['wrist'] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                print("Failed to read wrist camera")
                images['wrist'] = np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
        else:
            images['wrist'] = np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
        
        return images
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get full observation (state + images).
        
        Returns:
            Dict with keys matching GR00T's expected format:
                - 'observation.state': (7,) array
                - 'observation.images.external': (480, 640, 3) RGB
                - 'observation.images.wrist': (480, 640, 3) RGB
        """
        state = self.get_joint_states()
        images = self.get_camera_images()
        
        return {
            'observation.state': state,
            'observation.images.external': images['external'],
            'observation.images.wrist': images['wrist']
        }
    
    def go_home(self):
        """Move robot to home/safe position."""
        # Define a safe home position (customize for your setup)
        home_position = np.array([
            180.0,  # joint_1
            180.0,  # joint_2
            250.0,  # joint_3
            180.0,  # joint_4
            315.0,  # joint_5
            120.0,  # joint_6
            0.0     # gripper (open)
        ], dtype=np.float32)
        
        print("Moving to home position...")
        self.send_joint_positions(home_position, blocking=True)
        time.sleep(1.0)
        print("At home position")
    
    def emergency_stop(self):
        """Emergency stop - halt all motion."""
        if not self.is_connected:
            return
        
        try:
            print("!!! EMERGENCY STOP !!!")
            self.base.Stop()
        except Exception as e:
            print(f"Error during emergency stop: {e}")
