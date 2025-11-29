# Kinova API imports (replacing WidowX)
try:
    from kortex_api.TCPTransport import TCPTransport
    from kortex_api.RouterClient import RouterClient
    from kortex_api.SessionManager import SessionManager
    from kortex_api.autogen.messages import Base_pb2, DeviceConfig_pb2, Session_pb2
    from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
    KINOVA_AVAILABLE = True
    print("âœ… Kinova API imported successfully")
except AttributeError as e:
    if "MutableMapping" in str(e):
        print("âš ï¸  Kinova API has Python 3.10 compatibility issue")
        print("   This is a known issue with older Kinova API versions")
        print("   Basic functionality will still work")
        KINOVA_AVAILABLE = False
    else:
        raise e
except ImportError as e:
    print(f"âš ï¸  Kinova API not available: {e}")
    KINOVA_AVAILABLE = False

import time
import cv2



# Kinova Robot Controller Class
class KinovaController:
    """Kinova robot controller replacing WidowX functionality"""
    
    def __init__(self, ip="192.168.2.9", port=10000, credentials=("admin", "admin")):
        self.ip = ip
        self.port = port
        self.credentials = credentials
        
        # Connection objects
        self.transport = None
        self.router = None
        self.session_manager = None
        self.base = None
        
        # Movement parameters
        self.TIMEOUT_DURATION = 20
        
        print(f"Kinova Controller for BYOVLA")
        print(f"   IP: {self.ip}")
        print(f"   Port: {self.port}")
        print(f"   Credentials: {self.credentials}")
        print("=" * 40)
    
    def connect(self):
        """Connect to Kinova arm"""
        try:
            print("Connecting to Kinova robot...")
            
            # Set up API
            self.transport = TCPTransport()
            self.transport.connect(self.ip, self.port)
            self.router = RouterClient(self.transport)
            
            # Create session
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.credentials[0]
            session_info.password = self.credentials[1]
            session_info.session_inactivity_timeout = 60000
            session_info.connection_inactivity_timeout = 2000
            
            print("Creating session...")
            self.session_manager = SessionManager(self.router)
            session_handle = self.session_manager.CreateSession(session_info)
            
            # Create base client
            self.base = BaseClient(self.router)
            
            print("Connected to Kinova robot successfully!")
            return True
            
        except Exception as e:
            print(f"Failed to connect to Kinova robot: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Kinova arm"""
        try:
            if self.session_manager:
                self.session_manager.CloseSession()
            if self.transport:
                self.transport.disconnect()
            print("Disconnected from Kinova robot")
        except Exception as e:
            print(f"Error during disconnect: {e}")
    
    def get_current_pose(self):
        """Get current pose of the robot"""
        try:
            pose = self.base.GetMeasuredCartesianPose()
            pose_dict = {
                'x': pose.x,
                'y': pose.y,
                'z': pose.z,
                'theta_x': pose.theta_x,
                'theta_y': pose.theta_y,
                'theta_z': pose.theta_z
            }
            print(f"Retrieved pose: {pose_dict}")
            return pose_dict
        except Exception as e:
            print(f"Failed to get current pose: {e}")
            return None
    
    def is_connected(self):
        """Check if robot is connected and responsive"""
        try:
            # Try to get current pose as a connectivity test
            pose = self.base.GetMeasuredCartesianPose()
            return True
        except Exception as e:
            print(f"Robot connectivity test failed: {e}")
            return False
    
    def execute_action(self, dx, dy, dz, dtheta_x, dtheta_y, dtheta_z, duration=1.0):
        """Execute a relative movement action"""
        try:
            print(f"Executing movement: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}")
            print(f"Rotation: dtheta_x={dtheta_x:.3f}, dtheta_y={dtheta_y:.3f}, dtheta_z={dtheta_z:.3f}")
            
            # Get current pose
            current_pose = self.base.GetMeasuredCartesianPose()
            print(f"Current pose: x={current_pose.x:.3f}, y={current_pose.y:.3f}, z={current_pose.z:.3f}")
            
            # Calculate target pose
            target_x = current_pose.x + dx
            target_y = current_pose.y + dy
            target_z = current_pose.z + dz
            target_theta_x = current_pose.theta_x + dtheta_x
            target_theta_y = current_pose.theta_y + dtheta_y
            target_theta_z = current_pose.theta_z + dtheta_z
            
            print(f"Target pose: x={target_x:.3f}, y={target_y:.3f}, z={target_z:.3f}")
            
            # Try multiple movement methods
            success = False
            
            # Method 1: Try SendSelectedToolForConstrainedMotion
            try:
                print("Trying SendSelectedToolForConstrainedMotion...")
                target_pose = Base_pb2.Pose()
                target_pose.x = target_x
                target_pose.y = target_y
                target_pose.z = target_z
                target_pose.theta_x = target_theta_x
                target_pose.theta_y = target_theta_y
                target_pose.theta_z = target_theta_z
                
                self.base.SendSelectedToolForConstrainedMotion(target_pose)
                print("SendSelectedToolForConstrainedMotion executed")
                success = True
                
            except AttributeError as e:
                print(f"SendSelectedToolForConstrainedMotion not available: {e}")
            except Exception as e:
                print(f"SendSelectedToolForConstrainedMotion failed: {e}")
            
            # Method 2: Try ExecuteAction if Method 1 failed
            if not success:
                try:
                    print("Trying ExecuteAction...")
                    action = Base_pb2.Action()
                    action.reach_pose.target_pose.x = target_x
                    action.reach_pose.target_pose.y = target_y
                    action.reach_pose.target_pose.z = target_z
                    action.reach_pose.target_pose.theta_x = target_theta_x
                    action.reach_pose.target_pose.theta_y = target_theta_y
                    action.reach_pose.target_pose.theta_z = target_theta_z
                    
                    self.base.ExecuteAction(action)
                    print("ExecuteAction executed")
                    success = True
                    
                except Exception as e:
                    print(f"ExecuteAction failed: {e}")
            
            # Method 3: Try direct pose setting if both methods failed
            if not success:
                try:
                    print("Trying direct pose setting...")
                    # Try to set the pose directly
                    self.base.SetMeasuredCartesianPose(target_pose)
                    print("Direct pose setting executed")
                    success = True
                    
                except Exception as e:
                    print(f"Direct pose setting failed: {e}")
            
            if success:
                print(f"Waiting {duration} seconds for movement to complete...")
                time.sleep(duration)
                return True
            else:
                print("All movement methods failed")
                return False
            
        except Exception as e:
            print(f"Failed to execute action: {e}")
            return False
    
    def reset_to_home_position(self):
        """Reset the arm to a safe home position"""
        try:
            print("Resetting arm to home position...")
            
            # Define safe home position (adjust these values based on your setup)
            home_x = 0.4  # Forward position
            home_y = 0.0  # Center position
            home_z = 0.3  # Elevated position
            home_theta_x = -180.0  # Gripper pointing down
            home_theta_y = 0.0     # No pitch
            home_theta_z = 90.0    # Gripper oriented forward
            
            # Get current pose
            current_pose = self.base.GetMeasuredCartesianPose()
            
            # Calculate movement to home position
            dx = home_x - current_pose.x
            dy = home_y - current_pose.y
            dz = home_z - current_pose.z
            dtheta_x = home_theta_x - current_pose.theta_x
            dtheta_y = home_theta_y - current_pose.theta_y
            dtheta_z = home_theta_z - current_pose.theta_z
            
            print(f"Current: ({current_pose.x:.3f}, {current_pose.y:.3f}, {current_pose.z:.3f})")
            print(f"Target:  ({home_x:.3f}, {home_y:.3f}, {home_z:.3f})")
            print(f"Movement: ({dx:.3f}, {dy:.3f}, {dz:.3f})")
            
            # Execute movement to home position
            success = self.execute_action(dx, dy, dz, dtheta_x, dtheta_y, dtheta_z, 3.0)
            
            if success:
                print("Arm reset to home position successfully")
                return True
            else:
                print("Failed to reset arm to home position")
                return False
                
        except Exception as e:
            print(f"Error during arm reset: {e}")
            return False
    
    def move_to_absolute_position(self, x, y, z, theta_x, theta_y, theta_z, duration=3.0):
        """Move to an absolute position"""
        try:
            # Get current pose
            current_pose = self.base.GetMeasuredCartesianPose()
            
            # Calculate relative movement
            dx = x - current_pose.x
            dy = y - current_pose.y
            dz = z - current_pose.z
            dtheta_x = theta_x - current_pose.theta_x
            dtheta_y = theta_y - current_pose.theta_y
            dtheta_z = theta_z - current_pose.theta_z
            
            # Execute movement
            return self.execute_action(dx, dy, dz, dtheta_x, dtheta_y, dtheta_z, duration)
            
        except Exception as e:
            print(f"Failed to move to absolute position: {e}")
            return False
        
    def move_gripper(self, value):
        """
        Open or close the gripper
        :param value: value in [0, 1]. 0 for open, 1 for closed.
        :return: If operation was successful
        """
        self.gripper_timeout = 5 
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()
        # Set speed to open gripper
        print("Setting gripper position using velocity command...")
        gripper_command.mode = Base_pb2.GRIPPER_SPEED


        # Create message that will allow us to get feedback on gripper status
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
        current_value = gripper_measure.finger[0].value
        # Set velocity value depending on its sense (positive to open, negative to close. Yes, funny convention)
        # Close command
        if value > current_value:
            finger.value = -0.1
            self.base.SendGripperCommand(gripper_command)
            start = time.time()
            current_time = time.time()
            while current_time - start < self.gripper_timeout:
                gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
                current_value = gripper_measure.finger[0].value
                if current_value >= value:
                    return True
                current_time = time.time()
            return False
        # Open command
        if value < current_value:
            finger.value = 0.1
            self.base.SendGripperCommand(gripper_command)
            start = time.time()
            current_time = time.time()
            while current_time - start < self.gripper_timeout:
                gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
                current_value = gripper_measure.finger[0].value
                if current_value <= value:
                    return True
                current_time = time.time()
            return False
        
    def capture_image(self):

        camera_stream = f"rtsp://{self.ip}/color"
        cap = cv2.VideoCapture(camera_stream)
        timeout_sec = 3.0

        # Give the stream a moment to warm up
        start = time.time()
        frame = None
        while time.time() - start < timeout_sec:
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.05)

        cap.release()

        return frame if frame is not None else None