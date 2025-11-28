# Kinova Gen3 Deployment for GR00T

This directory contains scripts for deploying trained GR00T models on the Kinova Gen3 7-DOF robotic arm.

## Quick Start

### 1. Setup

Install Kinova Kortex API:
```bash
pip install kortex_api
```

Configure your robot's IP address in `config.py`.

### 2. Test Connection

```bash
python test_robot_connection.py
```

### 3. Deploy Model

```bash
python deploy_kinova.py \
    --checkpoint ./train_result/checkpoint-1000 \
    --task "pick up the orange cup and place it into the blue bowl" \
    --max_episodes 1
```

## Files

- `deploy_kinova.py` - Main deployment script
- `kinova_robot.py` - Robot control interface
- `config.py` - Configuration settings
- `test_robot_connection.py` - Test robot connectivity
- `README.md` - This file

## Safety

⚠️ **IMPORTANT SAFETY NOTES:**
- Always have emergency stop ready
- Start with conservative velocity/acceleration limits
- Test in open space away from obstacles
- Verify joint limits match your robot's configuration
- Monitor the robot during operation

## Troubleshooting

See the main repository documentation and Kinova Kortex API docs.
