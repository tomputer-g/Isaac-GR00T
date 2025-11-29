# Kinova Deployment Guide - Step by Step

## Prerequisites Checklist

- [ ] Robot powered on and connected to network
- [ ] Cameras connected (RealSense + Kinova wrist)
- [ ] Emergency stop accessible
- [ ] Clear workspace (no obstacles)
- [ ] Checkpoint trained and tested

## Step-by-Step Deployment

### 1. Test Components (IMPORTANT!)

```bash
# Test cameras only (no robot movement)
cd /home/ishita/L3D_Team4_src/Isaac-GR00T
python kinova_deployment/test_cameras_only.py
# ✓ Should show both cameras working

# Test robot connection (minimal movement to home position)
python kinova_deployment/test_robot_connection.py
# ✓ Should show joint states and save camera images
```

### 2. Start Policy Server (Terminal 1)

```bash
cd /home/ishita/L3D_Team4_src/Isaac-GR00T
python kinova_deployment/kinova_server.py --checkpoint train_result/checkpoint-5000
```

**Expected output:**
```
✓ Policy loaded successfully on cuda:0
Server is ready and listening on tcp://*:5555
```

Keep this terminal running!

### 3. Test Without Robot Movement (DRY RUN)

```bash
# Terminal 2: Test server connection without moving robot
python kinova_deployment/kinova_client.py \
    --task "Pick up the object" \
    --steps 5 \
    --dry_run
```

This will:
- ✓ Connect to server
- ✓ Connect to robot
- ✓ Get policy actions
- ✗ NOT send commands to robot

**Check:** You should see `[Dry run] Would send:` messages with action deltas.

### 4. VERY CONSERVATIVE First Run (Recommended!)

```bash
# Terminal 2: First real run with maximum safety
python kinova_deployment/kinova_client.py \
    --task "Pick up the object" \
    --steps 10 \
    --scale_actions 0.1 \
    --max_joint_delta 2.0 \
    --delay_per_action 0.2 \
    --interactive
```

**Safety settings explained:**
- `--scale_actions 0.1` = Only move 10% of commanded action (very conservative!)
- `--max_joint_delta 2.0` = Maximum 2° per action (very small movements)
- `--delay_per_action 0.2` = 5Hz control (slow)
- `--interactive` = Confirm each step manually

**You will see:**
```
[Step 0] Action preview:
  Joint deltas (degrees): [0.5, -0.3, 1.2, ...]
  Max delta: 1.2°
  Execute this action? [y/N/q]:
```

Type `y` to execute, `n` to skip, `q` to quit.

### 5. Progressive Testing

Once comfortable, gradually increase aggressiveness:

**Level 1: Very Safe (Start Here)**
```bash
python kinova_deployment/kinova_client.py \
    --task "Your task" \
    --steps 20 \
    --scale_actions 0.1 \
    --max_joint_delta 2.0 \
    --delay_per_action 0.2
```

**Level 2: Conservative**
```bash
python kinova_deployment/kinova_client.py \
    --task "Your task" \
    --steps 30 \
    --scale_actions 0.3 \
    --max_joint_delta 5.0 \
    --delay_per_action 0.1
```

**Level 3: Moderate**
```bash
python kinova_deployment/kinova_client.py \
    --task "Your task" \
    --steps 50 \
    --scale_actions 0.5 \
    --max_joint_delta 10.0 \
    --delay_per_action 0.05
```

**Level 4: Normal (Once Confident)**
```bash
python kinova_deployment/kinova_client.py \
    --task "Your task" \
    --steps 100 \
    --actions_per_step 4
```

### 6. Full Deployment with Video Recording

```bash
python kinova_deployment/kinova_client.py \
    --task "Pick up the red block and place it in the bowl" \
    --steps 100 \
    --save_video
```

Video will be saved as `kinova_execution_<timestamp>.mp4`

## Command Reference

### All Safety Options

```bash
python kinova_deployment/kinova_client.py \
    --host localhost \              # Server IP
    --port 5555 \                   # Server port
    --task "Task description" \     # What to do
    --steps 100 \                   # Number of policy steps
    --action_horizon 16 \           # Action chunk size (default: 16)
    --actions_per_step 4 \          # How many actions to execute per step
    --delay_per_action 0.05 \       # Time between actions (default: 0.05 = 20Hz)
    --scale_actions 0.3 \           # Scale movements (0.1 = very slow, 1.0 = full speed)
    --max_joint_delta 5.0 \         # Max degrees per action (safety limit)
    --interactive \                 # Confirm each step
    --dry_run \                     # Test without moving robot
    --save_video                    # Record execution
```

## Safety Recommendations

### Starting Out (Days 1-3)
- **Always use:** `--scale_actions 0.1` and `--max_joint_delta 2.0`
- **Always use:** `--steps 10` (short runs)
- **Consider using:** `--interactive` to approve each step
- Keep emergency stop within reach
- Start in open space, no obstacles

### Getting Confident (Days 4-7)
- Increase to: `--scale_actions 0.3` and `--max_joint_delta 5.0`
- Increase to: `--steps 30-50`
- Remove `--interactive` once comfortable
- Can work near objects but maintain safety margins

### Production Use (Week 2+)
- Use full speed: remove `--scale_actions` and `--max_joint_delta`
- Use full steps: `--steps 100+`
- Add `--save_video` for documentation
- Still keep emergency stop accessible

## Troubleshooting

### Robot moves too fast
```bash
# Slow it down!
--scale_actions 0.1 \
--max_joint_delta 2.0 \
--delay_per_action 0.2
```

### Robot doesn't move enough to complete task
```bash
# Gradually increase
--scale_actions 0.5 \  # was 0.1
--max_joint_delta 10.0  # was 2.0
```

### "Connection refused" error
```bash
# Make sure server is running first!
# Terminal 1:
python kinova_deployment/kinova_server.py --checkpoint <path>

# Then Terminal 2:
python kinova_deployment/kinova_client.py ...
```

### Robot jerks or unstable
```bash
# Use smoother control
--delay_per_action 0.1 \  # slower updates
--actions_per_step 2      # execute fewer actions per step
```

## Emergency Procedures

### If robot moves unexpectedly:
1. **Press emergency stop** (physical button on robot)
2. Press `Ctrl+C` in client terminal
3. Robot will attempt to return home before stopping

### If robot gets stuck:
1. Press `Ctrl+C` in client terminal
2. Wait for "Returning to home position..."
3. If it doesn't move, press emergency stop

### If you need to restart:
1. `Ctrl+C` in client terminal
2. `Ctrl+C` in server terminal (only if needed)
3. Power cycle robot if emergency stop was pressed
4. Start from Step 1 (test components)

## Example Session

```bash
# Terminal 1: Start server
python kinova_deployment/kinova_server.py --checkpoint train_result/checkpoint-5000

# Terminal 2: Safe first test
python kinova_deployment/kinova_client.py \
    --task "Pick up the cup" \
    --steps 10 \
    --scale_actions 0.1 \
    --interactive

# If successful, try less conservative
python kinova_deployment/kinova_client.py \
    --task "Pick up the cup" \
    --steps 20 \
    --scale_actions 0.3

# Once confident, full run
python kinova_deployment/kinova_client.py \
    --task "Pick up the cup" \
    --steps 100 \
    --save_video
```

## Quick Safety Checklist Before Each Run

- [ ] Emergency stop accessible
- [ ] Clear workspace
- [ ] Server running (Terminal 1)
- [ ] Starting with conservative settings
- [ ] Someone monitoring the robot
- [ ] Camera feeds visible

## Support

If issues persist:
1. Check `test_robot_connection.py` still works
2. Check server logs in Terminal 1
3. Try `--dry_run` to isolate server vs robot issues
4. Review camera images saved during test
