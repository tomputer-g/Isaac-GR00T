# Safe Testing Guide for Kinova Deployment

## 🛡️ Progressive Testing Strategy

Follow these tests **in order** from safest to full deployment.

---

## ✅ Level 0: NO Hardware Required

### Test 1: Policy Inference Only
**What it does:** Tests policy loading and inference with synthetic data  
**Hardware needed:** None (GPU/CPU only)  
**Risk:** Zero - no hardware interaction

```bash
cd kinova_deployment
python test_policy_only.py
```

**Expected output:**
- ✓ Checkpoint loads
- ✓ Policy runs inference
- ✓ Actions are validated
- Takes ~30-60 seconds (model loading)

**If it fails:**
- Check checkpoint path in `config.py`
- Verify CUDA/GPU availability
- Check that `datasets/kinova_dataset_nov6/meta/stats.json` exists

---

## ✅ Level 1: Cameras Only

### Test 2: Camera Access
**What it does:** Tests USB camera access and image capture  
**Hardware needed:** USB cameras only  
**Risk:** Very low - no robot movement

```bash
python test_cameras_only.py
```

**Expected output:**
- ✓ Both cameras detected
- ✓ Live feed displayed (10 seconds)
- ✓ Test images saved
- Shows FPS

**If it fails:**
- Check USB connections
- Update camera indices in `config.py`
- List cameras: `ls /dev/video*` (Linux) or Device Manager (Windows)

---

## ⚠️ Level 2: Robot Connection Only

### Test 3: Robot Connection (READ ONLY)
**What it does:** Connects and reads robot state  
**Hardware needed:** Kinova Gen3 robot  
**Risk:** Low - **NO COMMANDS SENT**

```bash
python test_robot_connection.py
```

**Expected output:**
- ✓ Robot connects
- ✓ Joint positions read
- ✓ Cameras work
- ✓ Full observation captured

**Safety:** This test only **reads** from the robot. It does NOT send any movement commands.

**If it fails:**
- Check robot IP in `config.py`
- Verify network connection: `ping 192.168.1.10`
- Check robot is powered on
- Verify kortex_api is installed: `pip install kortex_api`

---

## ⚠️⚠️ Level 3: Movement Commands (CAREFUL!)

### Test 4: Go Home Position
**What it does:** Sends robot to predefined home position  
**Hardware needed:** Kinova Gen3 robot  
**Risk:** Medium - **ROBOT WILL MOVE**

**Before running:**
1. ✅ Clear workspace of obstacles
2. ✅ Have emergency stop ready
3. ✅ Check home position in `kinova_robot.py` (line ~293)
4. ✅ Verify joint limits in `config.py`
5. ✅ Stand clear of robot

```python
# You can test home position from Python shell:
from kinova_robot import KinovaGen3Robot
robot = KinovaGen3Robot()
robot.connect()
# Robot will move to home!
robot.go_home()
robot.disconnect()
```

---

## ⚠️⚠️⚠️ Level 4: Full Deployment

### Test 5: Policy Deployment
**What it does:** Runs trained policy on real robot  
**Hardware needed:** Kinova Gen3 + cameras  
**Risk:** High - **AUTONOMOUS OPERATION**

**Before running:**
1. ✅ Complete ALL previous tests
2. ✅ Clear large workspace
3. ✅ Emergency stop accessible
4. ✅ Use soft/light objects
5. ✅ Continuous supervision

```bash
python deploy_kinova.py \
    --checkpoint ../train_result/checkpoint-1000 \
    --task "pick up the orange cup and place it into the blue bowl" \
    --max-episodes 1 \
    --visualize
```

**Emergency stop:**
- Press `Ctrl+C` in terminal
- Press `q` in visualization window
- Use physical E-stop button

---

## 📋 Complete Testing Checklist

Run these in order:

```bash
# 1. No hardware (SAFEST)
cd kinova_deployment
python test_policy_only.py           # ~60 seconds

# 2. Cameras only
python test_cameras_only.py          # ~15 seconds

# 3. Robot connection (read-only)
python test_robot_connection.py      # ~20 seconds

# 4. Robot movement (manual check)
# Check home position is safe first!

# 5. Full deployment (supervised)
python deploy_kinova.py \
    --checkpoint ../train_result/checkpoint-1000 \
    --task "your task here" \
    --max-episodes 1 \
    --visualize
```

---

## 🚨 Safety Reminders

### Before ANY robot movement:
- [ ] Workspace is clear (2+ meters)
- [ ] No people near robot
- [ ] Emergency stop accessible
- [ ] Camera feeds working
- [ ] Joint limits verified
- [ ] Velocity limits conservative

### During operation:
- [ ] Continuous visual monitoring
- [ ] Watch console for errors
- [ ] Ready to emergency stop
- [ ] Check for unexpected movements

### If something goes wrong:
1. **Immediate:** Press Ctrl+C or physical E-stop
2. **Check:** Console output for errors
3. **Verify:** Joint limits and safety margins
4. **Adjust:** Reduce velocity/acceleration in config
5. **Retest:** Start from safest test level

---

## 🎯 Recommended First-Time Workflow

Day 1: Software Only
```bash
python test_policy_only.py     # Verify policy works
```

Day 2: Cameras
```bash
python test_cameras_only.py    # Verify camera setup
```

Day 3: Robot Connection
```bash
python test_robot_connection.py  # Verify robot comms
```

Day 4: First Movement (with safety officer)
```bash
# Manual home position test
# Then try 1 episode with supervision
python deploy_kinova.py --max-episodes 1 --visualize
```

---

## 💡 Pro Tips

1. **Start slow:** Use `MAX_JOINT_VELOCITY = 10.0` in config first
2. **Test positions:** Verify home position is actually safe for YOUR setup
3. **Use soft objects:** Test with foam/soft items first
4. **Record video:** Helps debug issues
5. **Log everything:** Check console output carefully
6. **Iterate:** Adjust config based on observations

---

## ❓ Quick Troubleshooting

**Policy test fails?**
→ Check checkpoint path, verify GPU/CUDA

**Cameras fail?**
→ Update indices in config, check USB connections

**Robot connection fails?**
→ Ping robot IP, check network, verify kortex_api

**Actions out of range?**
→ Check stats.json, verify joint limits, retrain if needed

**Robot moves erratically?**
→ EMERGENCY STOP, check velocity limits, reduce control frequency

---

## ✅ Success Criteria

You're ready for deployment when:
- ✅ Policy test passes
- ✅ Cameras show clear images
- ✅ Robot connection stable
- ✅ Home position is safe
- ✅ Actions within limits
- ✅ Emergency stop tested
- ✅ Workspace is safe
- ✅ Supervision available

**Good luck and stay safe! 🤖**
