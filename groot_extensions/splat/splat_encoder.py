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
3D Gaussian Splat Scene Encoder

This module provides a simple PointNet-style encoder for encoding 3D Gaussian splat
scenes into a global embedding that can be used as additional context for GR00T.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SplatFeatureEncoder(nn.Module):
    """
    Simple PointNet-style encoder: per-splat MLP + global max pool.
    
    This encoder processes per-splat features through a two-layer MLP, applies
    global max pooling to obtain a permutation-invariant representation, and
    then projects to the final embedding dimension.
    
    Args:
        in_dim (int): Input dimension of per-splat features.
        hidden_dim (int): Hidden dimension for the MLPs. Default: 256.
        out_dim (int): Output dimension of the global scene embedding. Default: 128.
    
    Example:
        >>> encoder = SplatFeatureEncoder(in_dim=32, out_dim=128)
        >>> splat_feats = torch.randn(256, 32)  # 256 splats, 32 features each
        >>> z_3d = encoder(splat_feats)         # Output: (128,)
    """
    
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        self.mlp1 = nn.Linear(in_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.global_mlp = nn.Linear(hidden_dim, out_dim)

    def forward(self, splat_feats: torch.Tensor) -> torch.Tensor:
        """
        Encode per-splat features into a global scene embedding.
        
        Args:
            splat_feats (torch.Tensor): Per-splat features of shape (N, D_in),
                where N is the number of splats and D_in is the input feature dimension.
        
        Returns:
            torch.Tensor: Global scene embedding of shape (out_dim,).
        """
        # Per-splat feature processing
        x = F.relu(self.mlp1(splat_feats))      # (N, hidden_dim)
        x = F.relu(self.mlp2(x))                # (N, hidden_dim)
        
        # Global max pooling for permutation invariance
        x_pooled, _ = torch.max(x, dim=0)       # (hidden_dim,)
        
        # Project to final embedding dimension
        z_3d = self.global_mlp(x_pooled)        # (out_dim,)
        
        return z_3d


def build_dummy_splat_features(
    num_splats: int = 256,
    in_dim: int = 32,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Build dummy per-splat features for testing.
    
    This function generates random per-splat features sampled from a standard
    normal distribution. Useful for testing the encoder without actual 3D
    Gaussian splat data.
    
    Args:
        num_splats (int): Number of splats to generate. Default: 256.
        in_dim (int): Feature dimension per splat. Default: 32.
        device (str): Device to create the tensor on. Default: "cpu".
    
    Returns:
        torch.Tensor: Random splat features of shape (num_splats, in_dim).
    
    Example:
        >>> feats = build_dummy_splat_features(num_splats=512, in_dim=64)
        >>> feats.shape
        torch.Size([512, 64])
    """
    return torch.randn(num_splats, in_dim, device=device)


def make_dummy_splat_embedding(
    encoder: SplatFeatureEncoder,
    num_splats: int = 256,
    in_dim: int = 32,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a dummy splat embedding using random features.
    
    This is a convenience function that creates dummy splat features and
    encodes them into a global scene embedding. Useful for testing and
    prototyping.
    
    Args:
        encoder (SplatFeatureEncoder): The encoder to use.
        num_splats (int): Number of dummy splats to generate. Default: 256.
        in_dim (int): Feature dimension per splat. Default: 32.
        device (str): Device to run inference on. Default: "cpu".
    
    Returns:
        torch.Tensor: Global scene embedding of shape (out_dim,).
    
    Example:
        >>> encoder = SplatFeatureEncoder(in_dim=32, out_dim=128)
        >>> embedding = make_dummy_splat_embedding(encoder, num_splats=512)
        >>> embedding.shape
        torch.Size([128])
    """
    # Generate dummy features
    splat_feats = build_dummy_splat_features(
        num_splats=num_splats,
        in_dim=in_dim,
        device=device
    )
    
    # Move encoder to device and set to eval mode
    encoder = encoder.to(device)
    encoder.eval()
    
    # Generate embedding
    with torch.no_grad():
        z_3d = encoder(splat_feats)   # (out_dim,)
    
    return z_3d


if __name__ == "__main__":
    """
    Test the SplatFeatureEncoder with dummy data.
    """
    print("=" * 60)
    print("Testing SplatFeatureEncoder")
    print("=" * 60)
    
    # Configuration
    in_dim = 32
    hidden_dim = 256
    out_dim = 128
    num_splats = 256
    
    # Create encoder
    encoder = SplatFeatureEncoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim
    )
    
    print(f"\nEncoder Configuration:")
    print(f"  Input dimension:  {in_dim}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Output dimension: {out_dim}")
    print(f"  Number of splats: {num_splats}")
    
    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Total parameters: {num_params:,}")
    
    # Generate dummy embedding
    print(f"\nGenerating dummy embedding...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    z_3d = make_dummy_splat_embedding(
        encoder=encoder,
        num_splats=num_splats,
        in_dim=in_dim,
        device=device
    )
    
    print(f"\nOutput:")
    print(f"  Shape: {z_3d.shape}")
    print(f"  Dtype: {z_3d.dtype}")
    print(f"  Device: {z_3d.device}")
    print(f"  Min value: {z_3d.min().item():.4f}")
    print(f"  Max value: {z_3d.max().item():.4f}")
    print(f"  Mean value: {z_3d.mean().item():.4f}")
    print(f"  Std value: {z_3d.std().item():.4f}")
    
    # Test with different batch sizes
    print(f"\nTesting different numbers of splats:")
    for n in [64, 128, 512, 1024]:
        feats = build_dummy_splat_features(num_splats=n, in_dim=in_dim, device=device)
        encoder_dev = encoder.to(device)
        with torch.no_grad():
            emb = encoder_dev(feats)
        print(f"  {n:4d} splats -> embedding shape: {emb.shape}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
