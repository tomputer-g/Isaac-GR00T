# TROUBLESHOOTING: Robot Hitting Joint Limits

## Problem
Robot keeps hitting limits for joints 4 and 6, getting stuck during execution.

## Root Cause
The `JOINT_LIMITS` in `config.py` were set incorrectly - either too narrow or not matching your training data range.

## Solution Applied

### 1. Fixed Joint Limits in config.py

**OLD (Too Narrow):**
```python
JOINT_LIMITS = {
    'joint_3': (212.0, 296.0),  # Too narrow!
    'joint_5': (282.0, 349.0),  # Too narrow!
    'joint_6': (89.0, 144.0),   # Too narrow!
}
```

**NEW (Based on Your Training Data):**
```python
JOINT_LIMITS = {
    'joint_1': (0.0, 360.0),    # Full rotation
    'joint_2': (0.0, 360.0),    # Full rotation
    'joint_3': (192.0, 328.0),  # Expanded
    'joint_4': (0.0, 360.0),    # Full rotation
    'joint_5': (264.0, 360.0),  # Expanded
    'joint_6': (60.0, 200.0),   # Expanded
    'gripper': (0.0, 100.0)
}
```

### 2. Improved Client Joint Limit Handling

The client now:
- ✅ Checks each action against limits BEFORE sending
- ✅ Automatically clips to safe range
- ✅ Prints warnings when limits would be exceeded
- ✅ Tracks how many actions were successfully executed

## How to Verify the Fix

### Step 1: Check Your Training Data Limits

```bash
cd /home/ishita/L3D_Team4_src/Isaac-GR00T
python kinova_deployment/check_training_limits.py --dataset datasets/visible+bowl_36eps
```

This will show you the actual joint ranges from your training data.

### Step 2: Test with Conservative Settings

```bash
# Terminal 1: Server
python kinova_deployment/kinova_server.py --checkpoint train_result/checkpoint-5000

# Terminal 2: Client (safe test)
python kinova_deployment/kinova_client.py \
    --task "Pick and place" \
    --steps 20 \
    --scale_actions 0.2 \
    --dry_run
```

Check the output - you should see FEWER or NO limit warnings now.

### Step 3: Real Test (Still Conservative)

```bash
python kinova_deployment/kinova_client.py \
    --task "Pick and place" \
    --steps 30 \
    --scale_actions 0.2 \
    --max_joint_delta 5.0
```

Monitor the output:
```
  Step 0: Inference: 67.3ms, Executed: 4/4 actions  ← All actions executed!
  Step 10: Inference: 65.1ms, Executed: 4/4 actions
```

If you see:
```
  ⚠ joint_4 would exceed limits: 365.2° (range: 10.0-350.0°)
  Step 0: Inference: 67.3ms, Executed: 3/4 actions  ← Some actions skipped
```

This means the policy is still trying to move beyond training range.

## Why This Happens

1. **Training Data Range**: Your training demonstrations only covered certain joint ranges
2. **Policy Extrapolation**: The policy tries to command positions outside training range
3. **Safety System**: The robot's safety system rejects commands outside configured limits

## Solutions if Still Hitting Limits

### Option 1: Collect More Training Data
Cover a wider range of joint positions during data collection.

### Option 2: Increase Joint Limits (Carefully!)
If your robot physically CAN move beyond current limits:

```python
# In config.py, increase limits cautiously
JOINT_LIMITS = {
    'joint_6': (40.0, 220.0),  # Expanded from (60.0, 200.0)
}
```

⚠️ **Only do this if:**
- Robot physically supports the range
- You add safety margins
- You test incrementally

### Option 3: Use Action Scaling
Reduce action magnitude to stay within limits:

```bash
python kinova_deployment/kinova_client.py \
    --scale_actions 0.3 \  # 30% of commanded actions
    --max_joint_delta 5.0  # Max 5° per step
```

## Monitoring During Execution

Watch for these messages:

✅ **Good:**
```
  Step 10: Inference: 67.3ms, Executed: 4/4 actions
```
All actions executed successfully.

⚠️ **Warning:**
```
  ⚠ joint_4 would exceed limits: 365.2° (range: 10.0-350.0°)
  Step 10: Inference: 67.3ms, Executed: 3/4 actions
```
Some actions clipped, but execution continues.

❌ **Problem:**
```
  ⚠ joint_4 would exceed limits: 365.2° (range: 10.0-350.0°)
  ⚠ joint_6 would exceed limits: 205.3° (range: 70.0-190.0°)
  Step 10: Inference: 67.3ms, Executed: 1/4 actions
```
Many actions clipped, robot likely stuck.

## Emergency Recovery

If robot gets stuck in a bad position:

1. **Press Ctrl+C** - Client will try to return home
2. **If that fails** - Press emergency stop
3. **Restart**:
   ```bash
   # Power cycle robot if needed
   # Then test just going home:
   python kinova_deployment/test_robot_connection.py
   ```

## Recommended Deployment After Fix

```bash
# Terminal 1: Server
python kinova_deployment/kinova_server.py --checkpoint train_result/checkpoint-5000

# Terminal 2: Client (start conservative)
python kinova_deployment/kinova_client.py \
    --task "Your task" \
    --steps 50 \
    --scale_actions 0.3 \
    --save_video
```

The updated joint limits should allow smoother execution without hitting limits!
