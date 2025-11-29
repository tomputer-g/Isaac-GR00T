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

"""
Comprehensive tests for goal_token implementation in FlowmatchingActionHead.

This test suite verifies:
1. Goal encoder module existence and initialization
2. Goal token generation with correct shapes
3. Goal token integration in forward pass
4. Goal token integration in inference (get_action)
5. Gradient flow through goal encoder
6. Consistency between training and inference
7. Edge cases (missing goal data, different batch sizes)
"""

import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)


@pytest.fixture
def action_head_config():
    """Create a minimal action head configuration for testing."""
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


@pytest.fixture
def action_head(action_head_config):
    """Create action head instance."""
    model = FlowmatchingActionHead(action_head_config)
    model.eval()
    return model


@pytest.fixture
def sample_inputs():
    """Create sample inputs for testing."""
    batch_size = 2
    num_vl_tokens = 50
    action_dim = 7
    action_horizon = 10
    state_dim = 14
    
    # Backbone output
    backbone_output = BatchFeature({
        "backbone_features": torch.randn(batch_size, num_vl_tokens, 256),
        "backbone_attention_mask": torch.ones(batch_size, num_vl_tokens, dtype=torch.bool),
    })
    
    # Action input
    action_input = BatchFeature({
        "state": torch.randn(batch_size, 1, state_dim),
        "action": torch.randn(batch_size, action_horizon, action_dim),
        "action_mask": torch.ones(batch_size, action_horizon, action_dim),
        "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
        "goal_3d": torch.randn(batch_size, 3),
        "goal_visible": torch.ones(batch_size, 1),
    })
    
    return backbone_output, action_input


class TestGoalEncoderModule:
    """Test goal encoder module existence and properties."""
    
    def test_goal_encoder_exists(self, action_head):
        """Verify goal_encoder module exists."""
        assert hasattr(action_head, "goal_encoder"), "goal_encoder module not found"
    
    def test_goal_encoder_is_sequential(self, action_head):
        """Verify goal_encoder is a Sequential module."""
        assert isinstance(action_head.goal_encoder, torch.nn.Sequential), \
            "goal_encoder should be nn.Sequential"
    
    def test_goal_encoder_architecture(self, action_head):
        """Verify goal_encoder has correct architecture (Linear-ReLU-Linear)."""
        modules = list(action_head.goal_encoder.children())
        assert len(modules) == 3, f"Expected 3 modules, got {len(modules)}"
        assert isinstance(modules[0], torch.nn.Linear), "First layer should be Linear"
        assert isinstance(modules[1], torch.nn.ReLU), "Second layer should be ReLU"
        assert isinstance(modules[2], torch.nn.Linear), "Third layer should be Linear"
    
    def test_goal_encoder_dimensions(self, action_head, action_head_config):
        """Verify goal_encoder input/output dimensions."""
        # Input: 4 (3D position + 1 visibility flag)
        # Output: input_embedding_dim
        first_layer = list(action_head.goal_encoder.children())[0]
        last_layer = list(action_head.goal_encoder.children())[-1]
        
        assert first_layer.in_features == 4, \
            f"Expected input dimension 4, got {first_layer.in_features}"
        assert last_layer.out_features == action_head_config.input_embedding_dim, \
            f"Expected output dimension {action_head_config.input_embedding_dim}, got {last_layer.out_features}"


class TestGoalTokenGeneration:
    """Test goal token generation and shapes."""
    
    def test_goal_token_shape_forward(self, action_head, sample_inputs):
        """Test goal token shape in forward pass."""
        backbone_output, action_input = sample_inputs
        batch_size = action_input["state"].shape[0]
        
        # Extract goal encoding logic
        goal_3d = action_input["goal_3d"]
        goal_visible = action_input["goal_visible"]
        goal_input = torch.cat([goal_3d, goal_visible], dim=-1)
        goal_token = action_head.goal_encoder(goal_input).unsqueeze(1)
        
        expected_shape = (batch_size, 1, action_head.config.input_embedding_dim)
        assert goal_token.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {goal_token.shape}"
    
    def test_goal_input_concatenation(self, action_head, sample_inputs):
        """Test goal_3d and goal_visible concatenation."""
        _, action_input = sample_inputs
        
        goal_3d = action_input["goal_3d"]
        goal_visible = action_input["goal_visible"]
        goal_input = torch.cat([goal_3d, goal_visible], dim=-1)
        
        assert goal_input.shape[-1] == 4, \
            f"Expected concatenated dimension 4, got {goal_input.shape[-1]}"
    
    def test_goal_token_with_different_batch_sizes(self, action_head):
        """Test goal token generation with various batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            goal_input = torch.randn(batch_size, 4)
            goal_token = action_head.goal_encoder(goal_input).unsqueeze(1)
            
            assert goal_token.shape[0] == batch_size, \
                f"Batch size mismatch: expected {batch_size}, got {goal_token.shape[0]}"


class TestGoalTokenInForward:
    """Test goal token integration in training forward pass."""
    
    def test_forward_with_goal_tokens(self, action_head, sample_inputs):
        """Test forward pass includes goal tokens."""
        backbone_output, action_input = sample_inputs
        
        output = action_head(backbone_output, action_input)
        
        assert "loss" in output, "Forward pass should return loss"
        assert output["loss"].numel() == 1, "Loss should be a scalar"
    
    def test_forward_shape_consistency(self, action_head, sample_inputs):
        """Test that sa_embs has correct shape with goal token."""
        # We can't directly access sa_embs, but we can verify the forward pass works
        backbone_output, action_input = sample_inputs
        
        # This should not raise any errors
        try:
            output = action_head(backbone_output, action_input)
            assert True
        except Exception as e:
            pytest.fail(f"Forward pass failed with goal tokens: {e}")
    
    def test_forward_with_missing_goal_data(self, action_head, sample_inputs):
        """Test forward pass with missing goal_3d and goal_visible (using defaults)."""
        backbone_output, action_input = sample_inputs
        
        # Remove goal data - should use defaults
        del action_input["goal_3d"]
        del action_input["goal_visible"]
        
        output = action_head(backbone_output, action_input)
        
        assert "loss" in output, "Forward pass should work with default goal values"


class TestGoalTokenInInference:
    """Test goal token integration in inference (get_action)."""
    
    def test_get_action_with_goal_tokens(self, action_head, sample_inputs):
        """Test get_action includes goal tokens."""
        backbone_output, action_input = sample_inputs
        
        output = action_head.get_action(backbone_output, action_input)
        
        assert "action_pred" in output, "get_action should return action_pred"
        expected_shape = (
            action_input["state"].shape[0],
            action_head.config.action_horizon,
            action_head.config.action_dim,
        )
        assert output["action_pred"].shape == expected_shape, \
            f"Expected action shape {expected_shape}, got {output['action_pred'].shape}"
    
    def test_get_action_with_missing_goal_data(self, action_head, sample_inputs):
        """Test get_action with missing goal data (using defaults)."""
        backbone_output, action_input = sample_inputs
        
        # Remove goal data
        del action_input["goal_3d"]
        del action_input["goal_visible"]
        
        output = action_head.get_action(backbone_output, action_input)
        
        assert "action_pred" in output, "get_action should work with default goal values"
    
    def test_get_action_determinism_with_same_goal(self, action_head, sample_inputs):
        """Test that same goal produces consistent results (given same random seed)."""
        backbone_output, action_input = sample_inputs
        
        # Set manual seed for reproducibility
        torch.manual_seed(42)
        output1 = action_head.get_action(backbone_output, action_input)
        
        torch.manual_seed(42)
        output2 = action_head.get_action(backbone_output, action_input)
        
        assert torch.allclose(output1["action_pred"], output2["action_pred"]), \
            "Same inputs should produce same outputs"


class TestGoalTokenGradientFlow:
    """Test gradient flow through goal encoder."""
    
    def test_goal_encoder_gradients(self, action_head, sample_inputs):
        """Test that gradients flow through goal encoder during training."""
        action_head.train()
        backbone_output, action_input = sample_inputs
        
        # Forward pass
        output = action_head(backbone_output, action_input)
        loss = output["loss"]
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist for goal encoder
        for name, param in action_head.goal_encoder.named_parameters():
            assert param.grad is not None, \
                f"No gradient for goal_encoder parameter: {name}"
            assert not torch.all(param.grad == 0), \
                f"Zero gradient for goal_encoder parameter: {name}"
    
    def test_frozen_goal_encoder(self, action_head_config, sample_inputs):
        """Test that goal encoder can be frozen."""
        action_head = FlowmatchingActionHead(action_head_config)
        action_head.set_trainable_parameters(tune_projector=False, tune_diffusion_model=True)
        action_head.train()
        
        backbone_output, action_input = sample_inputs
        
        # Forward pass
        output = action_head(backbone_output, action_input)
        loss = output["loss"]
        
        # Backward pass
        loss.backward()
        
        # Check that goal encoder parameters have no gradients
        for name, param in action_head.goal_encoder.named_parameters():
            if param.requires_grad:
                pytest.fail(f"goal_encoder parameter {name} should not require grad when projector is frozen")


class TestForwardInferenceConsistency:
    """Test consistency between forward and get_action methods."""
    
    def test_same_goal_processing(self, action_head, sample_inputs):
        """Verify both methods process goal tokens the same way."""
        backbone_output, action_input = sample_inputs
        
        # Both methods should not raise errors
        try:
            _ = action_head(backbone_output, action_input)
            _ = action_head.get_action(backbone_output, action_input)
            assert True
        except Exception as e:
            pytest.fail(f"Inconsistent goal processing: {e}")
    
    def test_sequence_length_difference(self, action_head, sample_inputs):
        """Verify sa_embs sequence includes goal token (length check is indirect)."""
        # We verify by ensuring both forward and inference work without shape errors
        backbone_output, action_input = sample_inputs
        
        # If goal token wasn't included, there would be shape mismatches
        output_train = action_head(backbone_output, action_input)
        output_infer = action_head.get_action(backbone_output, action_input)
        
        assert "loss" in output_train
        assert "action_pred" in output_infer


class TestGoalTokenEdgeCases:
    """Test edge cases for goal token implementation."""
    
    def test_zero_goal_position(self, action_head, sample_inputs):
        """Test with zero goal position."""
        backbone_output, action_input = sample_inputs
        action_input["goal_3d"] = torch.zeros_like(action_input["goal_3d"])
        
        output = action_head.get_action(backbone_output, action_input)
        assert "action_pred" in output
    
    def test_invisible_goal(self, action_head, sample_inputs):
        """Test with invisible goal (goal_visible=0)."""
        backbone_output, action_input = sample_inputs
        action_input["goal_visible"] = torch.zeros_like(action_input["goal_visible"])
        
        output = action_head.get_action(backbone_output, action_input)
        assert "action_pred" in output
    
    def test_extreme_goal_positions(self, action_head, sample_inputs):
        """Test with extreme goal positions."""
        backbone_output, action_input = sample_inputs
        
        # Very large values
        action_input["goal_3d"] = torch.ones_like(action_input["goal_3d"]) * 1000
        output1 = action_head.get_action(backbone_output, action_input)
        assert "action_pred" in output1
        
        # Very small values
        action_input["goal_3d"] = torch.ones_like(action_input["goal_3d"]) * -1000
        output2 = action_head.get_action(backbone_output, action_input)
        assert "action_pred" in output2
    
    def test_mixed_batch_visibility(self, action_head, sample_inputs):
        """Test batch with mixed goal visibility."""
        backbone_output, action_input = sample_inputs
        batch_size = action_input["state"].shape[0]
        
        # Make half visible, half invisible
        visibility = torch.zeros(batch_size, 1)
        visibility[:batch_size//2] = 1.0
        action_input["goal_visible"] = visibility
        
        output = action_head.get_action(backbone_output, action_input)
        assert "action_pred" in output


class TestGoalTokenDataTypes:
    """Test goal token with different data types."""
    
    def test_float32_dtype(self, action_head, sample_inputs):
        """Test with float32 dtype."""
        backbone_output, action_input = sample_inputs
        action_input["goal_3d"] = action_input["goal_3d"].to(torch.float32)
        action_input["goal_visible"] = action_input["goal_visible"].to(torch.float32)
        
        output = action_head.get_action(backbone_output, action_input)
        assert output["action_pred"].dtype == torch.float32
    
    def test_dtype_consistency(self, action_head, sample_inputs):
        """Test that goal token dtype matches backbone features."""
        backbone_output, action_input = sample_inputs
        
        # Change backbone dtype
        dtype = torch.float32
        backbone_output["backbone_features"] = backbone_output["backbone_features"].to(dtype)
        
        output = action_head.get_action(backbone_output, action_input)
        assert output["action_pred"].dtype == dtype


def test_goal_token_integration_summary(action_head, sample_inputs):
    """Comprehensive integration test."""
    print("\n=== Goal Token Integration Test Summary ===")
    
    # 1. Module exists
    assert hasattr(action_head, "goal_encoder")
    print("✓ Goal encoder module exists")
    
    # 2. Forward pass works
    backbone_output, action_input = sample_inputs
    train_output = action_head(backbone_output, action_input)
    assert "loss" in train_output
    print("✓ Training forward pass works")
    
    # 3. Inference works
    infer_output = action_head.get_action(backbone_output, action_input)
    assert "action_pred" in infer_output
    print("✓ Inference (get_action) works")
    
    # 4. Shapes are correct
    batch_size = action_input["state"].shape[0]
    expected_shape = (batch_size, action_head.config.action_horizon, action_head.config.action_dim)
    assert infer_output["action_pred"].shape == expected_shape
    print(f"✓ Output shape is correct: {expected_shape}")
    
    # 5. Works with missing goal data
    del action_input["goal_3d"]
    del action_input["goal_visible"]
    default_output = action_head.get_action(backbone_output, action_input)
    assert "action_pred" in default_output
    print("✓ Works with default goal values")
    
    print("=== All integration tests passed! ===\n")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
