#!/usr/bin/env python
"""
Example usage of the SplatFeatureEncoder for encoding 3D Gaussian splat scenes.

This script demonstrates how to use the splat encoder with dummy features.
In a real application, you would replace the dummy features with actual
per-splat features extracted from a 3D Gaussian splatting scene.
"""

import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import torch
from groot_extensions.splat import (
    SplatFeatureEncoder,
    build_dummy_splat_features,
    make_dummy_splat_embedding,
)


def main():
    print("=" * 70)
    print("3D Gaussian Splat Scene Encoder - Example Usage")
    print("=" * 70)
    
    # Configuration
    in_dim = 32          # Feature dimension per splat (e.g., position + color + scale)
    hidden_dim = 256     # Hidden layer dimension
    out_dim = 128        # Output embedding dimension
    num_splats = 512     # Number of splats in the scene
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}\n")
    
    # ==================================================================
    # Method 1: Using the convenience function
    # ==================================================================
    print("Method 1: Using make_dummy_splat_embedding()")
    print("-" * 70)
    
    encoder = SplatFeatureEncoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    embedding = make_dummy_splat_embedding(
        encoder=encoder,
        num_splats=num_splats,
        in_dim=in_dim,
        device=device
    )
    
    print(f"Generated embedding:")
    print(f"  Shape: {embedding.shape}")
    print(f"  Device: {embedding.device}")
    print(f"  Mean: {embedding.mean().item():.4f}")
    print(f"  Std: {embedding.std().item():.4f}\n")
    
    # ==================================================================
    # Method 2: Manual step-by-step processing
    # ==================================================================
    print("Method 2: Manual step-by-step processing")
    print("-" * 70)
    
    # Step 1: Build or load per-splat features
    # In a real application, these would come from your 3D Gaussian splat scene
    # For example, concatenating [position(3), color(3), scale(3), rotation(4), ...]
    splat_features = build_dummy_splat_features(
        num_splats=num_splats,
        in_dim=in_dim,
        device=device
    )
    print(f"Splat features shape: {splat_features.shape}")
    
    # Step 2: Create and configure encoder
    encoder = SplatFeatureEncoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim
    ).to(device)
    
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters: {num_params:,}")
    
    # Step 3: Encode the scene
    encoder.eval()
    with torch.no_grad():
        scene_embedding = encoder(splat_features)
    
    print(f"Scene embedding shape: {scene_embedding.shape}")
    print(f"Scene embedding: {scene_embedding[:5].cpu().numpy()}... (showing first 5 values)\n")
    
    # ==================================================================
    # Method 3: Batch processing multiple scenes
    # ==================================================================
    print("Method 3: Processing multiple scenes with different splat counts")
    print("-" * 70)
    
    splat_counts = [128, 256, 512, 1024]
    embeddings = []
    
    for n_splats in splat_counts:
        feats = build_dummy_splat_features(num_splats=n_splats, in_dim=in_dim, device=device)
        with torch.no_grad():
            emb = encoder(feats)
        embeddings.append(emb)
        print(f"  {n_splats:4d} splats -> embedding shape: {emb.shape}, norm: {emb.norm().item():.4f}")
    
    # ==================================================================
    # Example: Integration with GR00T (conceptual)
    # ==================================================================
    print("\n" + "=" * 70)
    print("Conceptual Integration with GR00T")
    print("=" * 70)
    print("""
In a real GR00T integration, you might use the scene embedding as follows:

1. Extract per-splat features from your 3D Gaussian splat scene
2. Encode to global scene embedding using SplatFeatureEncoder
3. Concatenate or add scene embedding to GR00T's state representation
4. Use the augmented state for robot policy inference

Example pseudocode:
    
    # Get robot state and 3D scene
    robot_state = get_robot_state()           # e.g., (7,) for Kinova
    splat_features = get_splat_features()     # e.g., (N, 32)
    
    # Encode 3D scene
    scene_embedding = encoder(splat_features) # (128,)
    
    # Augment state with 3D scene context
    augmented_state = torch.cat([robot_state, scene_embedding])  # (7 + 128,)
    
    # Use with GR00T policy
    action = policy(
        observation={
            'video': video_obs,
            'state': augmented_state,  # Now includes 3D scene context!
            'language': task_description
        }
    )
    """)
    
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
