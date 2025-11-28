# Kinova Gen3 Deployment - Quick Start Guide

## Overview

This deployment package allows you to run trained GR00T models on a physical Kinova Gen3 7-DOF robotic arm in real-time.

---

## Setup Instructions

### 1. Install Dependencies

```bash
# Install Kinova Kortex API
pip install kortex_api

# Install OpenCV for camera handling
pip install opencv-python
```

### 2. Configure Your Robot

Edit `config.py` and set your robot's IP address:

```python
ROBOT_IP = "192.168.1.10"  # Change to your robot's IP
```

Also configure camera indices if needed:
```python
EXTERNAL_CAMERA_INDEX = 0  # USB camera for external view
WRIST_CAMERA_INDEX = 1     # USB camera for wrist view
```

### 3. Test Connection

Before running deployment, test that everything works:

```bash
cd kinova_deployment
python test_robot_connection.py
```

This will:
- ✓ Connect to the robot
- ✓ Read current joint positions
- ✓ Test camera feeds
- ✓ Display live camera views for 5 seconds

---

## Running Deployment

### Basic Usage

```bash
python deploy_kinova.py \
    --checkpoint ../train_result/checkpoint-1000 \
    --task "pick up the orange cup and place it into the blue bowl"
```

### Advanced Options

```bash
python deploy_kinova.py \
    --checkpoint <path_to_checkpoint> \
    --task <language_instruction> \
    --robot-ip 192.168.1.10 \
    --device cuda:0 \
    --max-episodes 5 \
    --visualize
```

**Arguments:**
- `--checkpoint`: Path to trained GR00T checkpoint (required)
- `--task`: Language description of the task (required)
- `--robot-ip`: Robot's IP address (default from config)
- `--device`: PyTorch device: `cuda:0` or `cpu`
- `--max-episodes`: Number of episodes to run
- `--visualize`: Show camera feed during execution

### Example Commands

**Single episode with visualization:**
```bash
python deploy_kinova.py \
    --checkpoint ../train_result/checkpoint-1000 \
    --task "pick up the orange cup and place it into the blue bowl" \
    --max-episodes 1 \
    --visualize
```

**Multiple episodes on CPU:**
```bash
python deploy_kinova.py \
    --checkpoint ../train_result/checkpoint-500 \
    --task "place the object in the basket" \
    --device cpu \
    --max-episodes 3
```

---

## File Structure

```
kinova_deployment/
├── README.md                    # This file
├── USAGE.md                     # Detailed usage guide
├── config.py                    # Configuration settings
├── kinova_robot.py              # Robot control interface
├── deploy_kinova.py             # Main deployment script
├── test_robot_connection.py    # Connection test utility
└── __init__.py                  # Package init
```

---

## How It Works

### 1. **Initialization**
- Connects to Kinova robot via Kortex API
- Initializes USB cameras (external + wrist views)
- Loads trained GR00T checkpoint
- Moves robot to safe home position

### 2. **Control Loop** (runs at 30 Hz)
```
┌─────────────────────────────────────┐
│  Capture Observation                │
│  - Joint positions (6 joints)       │
│  - Gripper position                 │
│  - External camera (640×480 RGB)    │
│  - Wrist camera (640×480 RGB)       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Preprocess for GR00T               │
│  - Resize images to 224×224         │
│  - Normalize state to [0,1]         │
│  - Apply transforms from training   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Policy Inference                   │
│  - Input: obs + language            │
│  - Output: 16 future actions        │
│           (action chunk)            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Denormalize Actions                │
│  - Convert [0,1] → degrees          │
│  - Apply safety limits              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Execute on Robot                   │
│  - Send 16 actions sequentially     │
│  - Each action: 6 joints + gripper  │
│  - Wait 1/30 sec between actions    │
└─────────────────────────────────────┘
```

### 3. **Safety Features**
- Joint limit checking with safety margins
- Automatic clipping of out-of-range commands
- Emergency stop on Ctrl+C or 'q' key
- Returns to home position after each episode

---

## Safety Guidelines

⚠️ **CRITICAL SAFETY INFORMATION:**

1. **Clear Workspace**
   - Remove obstacles from robot's reach
   - Ensure no people/objects in motion path
   - Have at least 2 meters clearance

2. **Emergency Stop**
   - Keep hand near keyboard
   - Press `Ctrl+C` or `q` to stop immediately
   - Physical E-stop button should be accessible

3. **Initial Testing**
   - Start with slow velocity limits (reduce `MAX_JOINT_VELOCITY` in config)
   - Test with soft/light objects first
   - Verify joint limits match your setup
   - Run test script before deployment

4. **Monitoring**
   - Always supervise during operation
   - Watch for unexpected movements
   - Check camera feeds are working
   - Monitor console for errors

5. **Joint Limits**
   - Default limits from training data:
     ```
     Joint 1: 0-360°
     Joint 2: 0-360°
     Joint 3: 212-296°  (limited range!)
     Joint 4: 0-360°
     Joint 5: 282-349°  (limited range!)
     Joint 6: 89-144°   (limited range!)
     Gripper: 0-100
     ```
   - **Verify these match YOUR robot's configuration!**

---

## Troubleshooting

### Robot Connection Issues

**Problem:** `Failed to connect to robot`
**Solution:**
- Verify robot IP: `ping 192.168.1.10`
- Check network connection
- Ensure robot is powered on
- Try different port (default: 10000)

### Camera Not Found

**Problem:** `Failed to setup cameras`
**Solution:**
- Check USB camera connections
- List devices: `ls /dev/video*` (Linux) or check Device Manager (Windows)
- Update camera indices in `config.py`
- Test with: `python -c "import cv2; print(cv2.VideoCapture(0).read())"`

### Out of Memory

**Problem:** CUDA out of memory error
**Solution:**
- Use CPU: `--device cpu`
- Close other GPU programs
- Reduce batch size (modify code if needed)

### Actions Out of Range

**Problem:** `WARNING: Target position outside safe limits`
**Solution:**
- Check joint limit configuration in `config.py`
- Increase `SAFETY_MARGIN`
- Retrain model with correct joint ranges
- Verify stats.json matches your robot

### Slow Performance

**Problem:** FPS < 30 Hz
**Solution:**
- Use GPU: `--device cuda:0`
- Close visualization: remove `--visualize`
- Check CPU usage
- Reduce camera resolution in config

---

## Customization

### Change Control Frequency

Edit `config.py`:
```python
CONTROL_FREQUENCY = 20  # Hz (default: 30)
```

### Modify Home Position

Edit `kinova_robot.py`, method `go_home()`:
```python
home_position = np.array([
    180.0,  # joint_1
    180.0,  # joint_2
    250.0,  # joint_3
    180.0,  # joint_4
    315.0,  # joint_5
    120.0,  # joint_6
    0.0     # gripper (open)
])
```

### Add Custom Safety Checks

In `kinova_robot.py`, modify `_is_safe_position()`:
```python
def _is_safe_position(self, positions: np.ndarray) -> bool:
    # Add your custom checks here
    # Return False if position is unsafe
    return True
```

---

## Next Steps

1. ✓ Test connection: `python test_robot_connection.py`
2. ✓ Run single episode: `python deploy_kinova.py --checkpoint <path> --task <task>`
3. ✓ Monitor and adjust safety limits as needed
4. ✓ Deploy for real tasks

---

## Support

For issues or questions:
- Check repository documentation
- Review GR00T paper and examples
- Consult Kinova Kortex API docs
- Check SO-100 deployment example in `examples/SO-100/`

**Good luck with your deployment! 🤖**
