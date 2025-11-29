# Kinova Deployment - Quick Start Card

## 🚀 DEPLOYMENT IN 3 STEPS

### 1️⃣ Start Server (Terminal 1)
```bash
cd /home/ishita/L3D_Team4_src/Isaac-GR00T
python kinova_deployment/kinova_server.py --checkpoint train_result/checkpoint-5000
```

### 2️⃣ Test Without Moving (Terminal 2)
```bash
python kinova_deployment/kinova_client.py --task "Your task" --steps 5 --dry_run
```

### 3️⃣ VERY SAFE First Run
```bash
python kinova_deployment/kinova_client.py \
    --task "Your task" \
    --steps 10 \
    --scale_actions 0.1 \
    --max_joint_delta 2.0 \
    --interactive
```

---

## 🎚️ SAFETY LEVELS

### 🐢 VERY SAFE (Start here!)
```bash
--scale_actions 0.1 --max_joint_delta 2.0 --delay_per_action 0.2 --interactive
```
- Moves at 10% speed
- Max 2° per action
- Asks permission each step

### 🚶 CONSERVATIVE
```bash
--scale_actions 0.3 --max_joint_delta 5.0 --delay_per_action 0.1
```
- Moves at 30% speed
- Max 5° per action

### 🏃 NORMAL (After testing)
```bash
--steps 100 --save_video
```
- Full speed
- Record video

---

## ⚙️ KEY PARAMETERS

| Parameter | What it does | Safe value | Normal |
|-----------|-------------|------------|--------|
| `--scale_actions` | Speed multiplier | `0.1` | `1.0` |
| `--max_joint_delta` | Max angle change | `2.0` | None |
| `--delay_per_action` | Time between moves | `0.2s` | `0.05s` |
| `--steps` | How many steps | `10` | `100` |
| `--interactive` | Ask before each step | Yes | No |
| `--dry_run` | Don't move robot | For testing | No |

---

## 🛑 EMERGENCY

**Robot moving wrong:** Press emergency stop → `Ctrl+C`

**Need to stop:** `Ctrl+C` → waits for home → disconnects

---

## ✅ PRE-FLIGHT CHECKLIST

- [ ] Server running (Terminal 1)
- [ ] Emergency stop ready
- [ ] Starting with `--scale_actions 0.1`
- [ ] Clear workspace
- [ ] Someone monitoring

---

## 📖 Full Guide

See `DEPLOYMENT_GUIDE.md` for detailed instructions.
