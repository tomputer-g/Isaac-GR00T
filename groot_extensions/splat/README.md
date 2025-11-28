# GR00T Extensions: 3D Gaussian Splat Encoder

This module provides a simple encoder for converting 3D Gaussian splat scenes into global embeddings that can be used as additional context for GR00T.

## Overview

The `SplatFeatureEncoder` is a PointNet-style architecture that:
1. Processes per-splat features through a two-layer MLP
2. Applies global max pooling for permutation invariance
3. Projects to a final embedding dimension

## Usage

### Basic Usage

```python
from groot_extensions.splat import SplatFeatureEncoder, make_dummy_splat_embedding

# Create encoder
encoder = SplatFeatureEncoder(in_dim=32, hidden_dim=256, out_dim=128)

# Generate a dummy embedding for testing
embedding = make_dummy_splat_embedding(encoder, num_splats=256)
print(f"Embedding shape: {embedding.shape}")  # Output: torch.Size([128])
```

### With Real Splat Features

```python
import torch
from groot_extensions.splat import SplatFeatureEncoder

# Assuming you have per-splat features (e.g., position, color, scale, rotation)
# Shape: (N, feature_dim) where N is the number of splats
splat_features = torch.randn(512, 32)  # 512 splats, 32 features each

# Create and use encoder
encoder = SplatFeatureEncoder(in_dim=32, out_dim=128)
encoder.eval()

with torch.no_grad():
    scene_embedding = encoder(splat_features)  # Output: (128,)
```

## Architecture

```
Input: (N, in_dim) per-splat features
  ↓
MLP1: Linear(in_dim → hidden_dim) + ReLU
  ↓
MLP2: Linear(hidden_dim → hidden_dim) + ReLU
  ↓
Global Max Pool: (N, hidden_dim) → (hidden_dim,)
  ↓
Global MLP: Linear(hidden_dim → out_dim)
  ↓
Output: (out_dim,) global scene embedding
```

## Parameters

- **in_dim**: Input feature dimension per splat (e.g., 32)
- **hidden_dim**: Hidden layer dimension (default: 256)
- **out_dim**: Output embedding dimension (default: 128)

## API Reference

### `SplatFeatureEncoder`

```python
SplatFeatureEncoder(in_dim: int, hidden_dim: int = 256, out_dim: int = 128)
```

Main encoder class that processes splat features into a global scene embedding.

### `build_dummy_splat_features`

```python
build_dummy_splat_features(
    num_splats: int = 256,
    in_dim: int = 32,
    device: str = "cpu"
) -> torch.Tensor
```

Generate random splat features for testing.

### `make_dummy_splat_embedding`

```python
make_dummy_splat_embedding(
    encoder: SplatFeatureEncoder,
    num_splats: int = 256,
    in_dim: int = 32,
    device: str = "cpu"
) -> torch.Tensor
```

Convenience function to generate a dummy scene embedding.

## Testing

Run the built-in test:

```bash
python groot_extensions/splat/splat_encoder.py
```

This will test the encoder with various numbers of splats and display statistics about the generated embeddings.

## Future Enhancements

- [ ] Support for batch processing of multiple scenes
- [ ] Attention-based pooling instead of max pooling
- [ ] Pre-trained weights for common splat feature types
- [ ] Integration with 3D Gaussian splatting rendering pipelines
- [ ] Multi-scale feature extraction
- [ ] Learnable positional encodings for spatial information
