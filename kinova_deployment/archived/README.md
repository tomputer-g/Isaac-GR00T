# Kinova GR00T Deployment - Server-Client Architecture

This directory contains the **proper GR00T deployment architecture** for the Kinova Gen3 robot, following NVIDIA's recommended client-server pattern.

## ⚠️ Important: Architecture Change

**Your original `deploy_kinova.py` does NOT follow GR00T's design!**

GR00T uses a **client-server architecture** (see `examples/SO-100/eval_gr00t_so100.py`):
- ❌ **Old Way**: `deploy_kinova.py` - Everything in one script (not how GR00T works)
- ✅ **New Way**: `kinova_server.py` + `kinova_client.py` - Proper separation

## Architecture Overview

```
┌─────────────────────────────────────────┐
│   GPU Machine (Server)                  │
│   kinova_server.py                      │
│   - Loads policy checkpoint             │
│   - Runs inference on GPU               │
│   - ZeroMQ server on port 5555          │
└─────────────────┬───────────────────────┘
                  │
                  │ ZeroMQ (tcp)
                  │
┌─────────────────┴───────────────────────┐
│   Robot Machine (Client)                │
│   kinova_client.py                      │
│   - Connects to robot + cameras         │
│   - Sends observations to server        │
│   - Receives actions                    │
│   - Executes on robot                   │
└─────────────────────────────────────────┘
```

## Quick Start

### Step 1: Start Policy Server (Terminal 1)

```bash
cd /home/ishita/L3D_Team4_src/Isaac-GR00T
python kinova_deployment/kinova_server.py --checkpoint train_result/checkpoint-5000
```

Output:
```
✓ Policy loaded successfully on cuda:0
Server is ready and listening on tcp://*:5555
```

### Step 2: Run Robot Client (Terminal 2)

```bash
cd /home/ishita/L3D_Team4_src/Isaac-GR00T
python kinova_deployment/kinova_client.py --task "Pick up the object"
```

## Files

### ⭐ NEW: Server-Client (Use These!)

- **`kinova_server.py`** - Policy inference server (runs on GPU machine)
- **`kinova_client.py`** - Robot control client (runs on robot/same machine)

### Core Infrastructure

- **`kinova_robot.py`** - Kinova Gen3 interface (Kortex API + cameras)
- **`config.py`** - Configuration (IPs, ports, camera indices)

### Testing Scripts

- **`test_robot_connection.py`** - Test robot + cameras ✅ (working)
- **`test_cameras_only.py`** - Test cameras only ✅ (working)
- **`test_policy_only.py`** - Test policy inference ✅ (working)

### Legacy (Don't Use)

- **`deploy_kinova.py`** - Old monolithic script (doesn't follow GR00T pattern)
- **`deploy.py`** - Old file

## Configuration

Edit `config.py`:

```python
ROBOT_IP = "192.168.2.9"          # Your Kinova IP
POLICY_SERVER_HOST = "localhost"   # Or GPU machine IP
POLICY_SERVER_PORT = 5555
```

## Command Line Options

### Server Options

```bash
python kinova_deployment/kinova_server.py \
    --checkpoint train_result/checkpoint-5000 \  # Required
    --port 5555 \                                 # Default: 5555
    --device cuda:0                               # Default: cuda:0
```

### Client Options

```bash
python kinova_deployment/kinova_client.py \
    --host localhost \              # Server IP (default: localhost)
    --port 5555 \                   # Server port (default: 5555)
    --task "Your task" \            # Task description
    --steps 100 \                   # Number of steps (default: 100)
    --action_horizon 16 \           # Action horizon (default: 16)
    --actions_per_step 4 \          # Actions to execute per step
    --save_video                    # Save execution video
```

## Testing Before Deployment

```bash
# 1. Test cameras
python kinova_deployment/test_cameras_only.py
# Should show: ✓ Both cameras working

# 2. Test robot connection  
python kinova_deployment/test_robot_connection.py
# Should show: ✓ All tests passed

# 3. Test policy (no robot)
python kinova_deployment/test_policy_only.py
# Should show: ✓ Policy returns actions (16, 7)
```

## Example Workflows

### Same Machine (Server + Client Together)

```bash
# Terminal 1: Server
python kinova_deployment/kinova_server.py --checkpoint train_result/checkpoint-5000

# Terminal 2: Client
python kinova_deployment/kinova_client.py --task "Pick and place" --steps 50
```

### Different Machines (Server on GPU, Client on Robot)

```bash
# On GPU machine (IP: 192.168.1.100)
python kinova_deployment/kinova_server.py --checkpoint train_result/checkpoint-5000

# On robot machine
python kinova_deployment/kinova_client.py \
    --host 192.168.1.100 \
    --task "Pick and place" \
    --steps 50
```

## Why This Architecture?

Matches NVIDIA's design (`gr00t/eval/robot.py`, `examples/SO-100/`):

| Aspect | Old deploy_kinova.py | New server+client |
|--------|---------------------|-------------------|
| **Follows GR00T pattern** | ❌ No | ✅ Yes |
| **GPU flexibility** | Must be on robot | Can be remote |
| **Multi-robot** | No | Yes (one server, many clients) |
| **Testing** | Hard | Easy (test components separately) |
| **Network overhead** | N/A | ~1-2ms (minimal) |

## Troubleshooting

### "Connection refused"

```bash
# Make sure server is running FIRST
python kinova_deployment/kinova_server.py --checkpoint <path>

# Then start client
python kinova_deployment/kinova_client.py
```

### Black camera images

RTSP stream needs 5-10s warmup (already handled in code). If still black:

```bash
# Test cameras separately
python kinova_deployment/test_cameras_only.py
```

### "Module not found: gr00t"

```bash
cd /home/ishita/L3D_Team4_src/Isaac-GR00T
pip install -e .
```

## Performance

- **Inference latency**: 50-100ms (GPU dependent)
- **Control frequency**: 20Hz (50ms per action)
- **Action horizon**: 16 (typically execute 4-8 per step)
- **Network overhead**: ~1-2ms on local network

## Safety Notes

⚠️ **Always:**
- Have emergency stop ready
- Start with few steps (--steps 10)
- Monitor robot during execution
- Test in open space
- Verify joint limits in config.py

## References

- GR00T SO-100 example: `examples/SO-100/eval_gr00t_so100.py`
- Server API: `gr00t/eval/robot.py` (RobotInferenceServer/Client)
- Your eval script: `l3d_src/kinova_eval.py` (uses same pattern)
