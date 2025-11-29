#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Quick validation script for goal_token implementation.
Run this to do a manual sanity check of the goal token integration.
"""

import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)


def create_dummy_config():
    """Create minimal config for testing."""
    return FlowmatchingActionHeadConfig(
        hidden_size=128,
        input_embedding_dim=256,
        backbone_embedding_dim=256,
        action_dim=7,
        action_horizon=10,
        max_state_dim=14,
        max_num_embodiments=5,
        num_timestep_buckets=100,
        num_inference_timesteps=10,
        add_pos_embed=True,
        max_seq_len=512,
        num_target_vision_tokens=32,
        use_vlln=False,
        diffusion_model_cfg={
            "input_size": 256,
            "hidden_size": 128,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 4.0,
        },
    )


def create_dummy_inputs(batch_size=2):
    """Create dummy inputs."""
    backbone_output = BatchFeature({
        "backbone_features": torch.randn(batch_size, 50, 256),
        "backbone_attention_mask": torch.ones(batch_size, 50, dtype=torch.bool),
    })
    
    action_input = BatchFeature({
        "state": torch.randn(batch_size, 1, 14),
        "action": torch.randn(batch_size, 10, 7),
        "action_mask": torch.ones(batch_size, 10, 7),
        "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
        "goal_3d": torch.randn(batch_size, 3),
        "goal_visible": torch.ones(batch_size, 1),
    })
    
    return backbone_output, action_input


def test_module_existence(model):
    """Test 1: Check if goal_encoder exists."""
    print("\n[Test 1] Checking goal_encoder module...")
    if hasattr(model, "goal_encoder"):
        print("  ✓ goal_encoder module exists")
        print(f"  ✓ Architecture: {model.goal_encoder}")
        return True
    else:
        print("  ✗ goal_encoder module NOT found!")
        return False


def test_goal_token_shape(model):
    """Test 2: Check goal token shape."""
    print("\n[Test 2] Checking goal token shape...")
    batch_size = 3
    goal_input = torch.randn(batch_size, 4)
    
    try:
        goal_token = model.goal_encoder(goal_input).unsqueeze(1)
        expected_shape = (batch_size, 1, model.config.input_embedding_dim)
        
        if goal_token.shape == expected_shape:
            print(f"  ✓ Goal token shape correct: {goal_token.shape}")
            return True
        else:
            print(f"  ✗ Shape mismatch! Expected {expected_shape}, got {goal_token.shape}")
            return False
    except Exception as e:
        print(f"  ✗ Error generating goal token: {e}")
        return False


def test_forward_pass(model, backbone_output, action_input):
    """Test 3: Test forward pass with goal tokens."""
    print("\n[Test 3] Testing forward pass...")
    try:
        model.train()
        output = model(backbone_output, action_input)
        
        if "loss" in output:
            print(f"  ✓ Forward pass successful, loss: {output['loss'].item():.4f}")
            return True
        else:
            print("  ✗ Forward pass missing loss!")
            return False
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_pass(model, backbone_output, action_input):
    """Test 4: Test inference (get_action) with goal tokens."""
    print("\n[Test 4] Testing inference (get_action)...")
    try:
        model.eval()
        output = model.get_action(backbone_output, action_input)
        
        if "action_pred" in output:
            expected_shape = (
                action_input["state"].shape[0],
                model.config.action_horizon,
                model.config.action_dim,
            )
            actual_shape = output["action_pred"].shape
            
            if actual_shape == expected_shape:
                print(f"  ✓ Inference successful, action shape: {actual_shape}")
                return True
            else:
                print(f"  ✗ Shape mismatch! Expected {expected_shape}, got {actual_shape}")
                return False
        else:
            print("  ✗ Inference missing action_pred!")
            return False
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_default_goal_values(model, backbone_output, action_input):
    """Test 5: Test with missing goal data (should use defaults)."""
    print("\n[Test 5] Testing with missing goal data (defaults)...")
    try:
        # Remove goal data
        action_input_copy = BatchFeature(dict(action_input))
        if "goal_3d" in action_input_copy:
            del action_input_copy["goal_3d"]
        if "goal_visible" in action_input_copy:
            del action_input_copy["goal_visible"]
        
        model.eval()
        output = model.get_action(backbone_output, action_input_copy)
        
        if "action_pred" in output:
            print("  ✓ Works with default goal values")
            return True
        else:
            print("  ✗ Failed with default goal values")
            return False
    except Exception as e:
        print(f"  ✗ Failed with defaults: {e}")
        return False


def test_gradient_flow(model, backbone_output, action_input):
    """Test 6: Test gradient flow through goal_encoder."""
    print("\n[Test 6] Testing gradient flow...")
    try:
        model.train()
        
        # Forward + backward
        output = model(backbone_output, action_input)
        loss = output["loss"]
        loss.backward()
        
        # Check gradients
        has_gradients = False
        for name, param in model.goal_encoder.named_parameters():
            if param.grad is not None and not torch.all(param.grad == 0):
                has_gradients = True
                break
        
        if has_gradients:
            print("  ✓ Gradients flow through goal_encoder")
            return True
        else:
            print("  ✗ No gradients in goal_encoder!")
            return False
    except Exception as e:
        print(f"  ✗ Gradient test failed: {e}")
        return False


def test_frozen_goal_encoder():
    """Test 7: Test frozen goal_encoder."""
    print("\n[Test 7] Testing frozen goal_encoder...")
    try:
        config = create_dummy_config()
        model = FlowmatchingActionHead(config)
        model.set_trainable_parameters(tune_projector=False, tune_diffusion_model=True)
        
        # Check if goal_encoder is frozen
        all_frozen = all(not p.requires_grad for p in model.goal_encoder.parameters())
        
        if all_frozen:
            print("  ✓ goal_encoder correctly frozen when tune_projector=False")
            return True
        else:
            print("  ✗ goal_encoder not properly frozen!")
            return False
    except Exception as e:
        print(f"  ✗ Frozen test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Goal Token Implementation Validation")
    print("=" * 60)
    
    # Create model and inputs
    print("\nInitializing model...")
    config = create_dummy_config()
    model = FlowmatchingActionHead(config)
    backbone_output, action_input = create_dummy_inputs()
    print("✓ Model and inputs created")
    
    # Run tests
    results = []
    results.append(("Module Existence", test_module_existence(model)))
    results.append(("Goal Token Shape", test_goal_token_shape(model)))
    results.append(("Forward Pass", test_forward_pass(model, backbone_output, action_input)))
    results.append(("Inference Pass", test_inference_pass(model, backbone_output, action_input)))
    results.append(("Default Values", test_default_goal_values(model, backbone_output, action_input)))
    results.append(("Gradient Flow", test_gradient_flow(model, backbone_output, action_input)))
    results.append(("Frozen Encoder", test_frozen_goal_encoder()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} | {test_name}")
    
    print("-" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Goal token implementation looks correct.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())
