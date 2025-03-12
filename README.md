# PyTorch Flash Rotary Attention

This repository implements an efficient attention mechanism combining several modern techniques:
- Flash Attention for memory-efficient attention computation
- Rotary Positional Embeddings (RoPE) for better position-aware attention
- Sliding Window Attention for processing long sequences
- Grouped Query Attention for improved efficiency

## Features

- **Flash Attention**: Utilizes PyTorch's native flash attention implementation for efficient memory usage
- **Rotary Embeddings**: Implements RoPE for better handling of positional information
- **Sliding Window**: Processes long sequences by splitting them into overlapping windows
- **Grouped Query Attention**: Reduces computation by sharing key-value heads across multiple query heads

## Installation

Requirements:
- Python 3.7+
- PyTorch 2.0+
- einops

```bash
pip install torch einops
```

## Basic Usage

```python
import torch
from attention import SlidingWindowAttention

# Initialize the attention module
attn = SlidingWindowAttention(
    dim=512,           # Model dimension
    heads=8,           # Number of attention heads
    dim_head=64,       # Dimension per head
    window_size=512,   # Size of sliding window
    dropout=0.1        # Dropout rate
)

# Create input tensor
x = torch.randn(1, 1024, 512)  # [batch_size, sequence_length, dimension]

# Get attention output
output = attn(x)
```

## Examples

The repository includes several examples demonstrating different use cases:

### 1. Basic Flash Attention
```python
from examples.basic_usage import test_flash_attention

# Test basic flash attention with standard sequence length
test_flash_attention()
```

### 2. Sliding Window Attention for Long Sequences
```python
from examples.basic_usage import test_sliding_window_attention

# Test attention with long sequences using sliding window
test_sliding_window_attention()
```

### 3. Transformer Block Implementation
```python
from examples.basic_usage import test_transformer_block

# Test attention in a transformer block context
test_transformer_block()
```

### 4. Text Classification Example
```python
from examples.basic_usage import test_text_classifier

# Test attention for text classification
test_text_classifier()
```

### 5. Vision Models
The repository includes examples of using the attention mechanism in vision models:

#### Vision Transformer (ViT)
```python
from examples.vision_models import SimpleViT

# Initialize ViT with Flash Attention
vit = SimpleViT(
    img_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    heads=12
)

# Process image
img = torch.randn(1, 3, 224, 224)
output = vit(img)
```

#### Hybrid CNN-Attention
```python
from examples.vision_models import HybridCNNAttention

# Initialize hybrid model
model = HybridCNNAttention(
    in_channels=3,
    num_classes=1000
)

# Process image
img = torch.randn(1, 3, 224, 224)
output = model(img)
```

The vision models demonstrate two approaches to incorporating attention:
1. **Pure Transformer (ViT)**: Processes images using only attention mechanisms
2. **Hybrid CNN-Attention**: Combines convolutional layers with attention for feature refinement

To run all examples:
```bash
python examples/basic_usage.py
python examples/vision_models.py
```

## Architecture Details

### RotaryEmbedding
- Implements rotary positional embeddings
- Uses frequency-based position encoding
- Applies rotation to query and key vectors

### FlashAttention
- Base attention implementation using PyTorch's scaled_dot_product_attention
- Incorporates rotary embeddings
- Supports grouped query attention
- Uses flash attention when available on the hardware

### SlidingWindowAttention
- Extends FlashAttention with windowing mechanism
- Processes long sequences in overlapping windows
- Automatically falls back to regular attention for short sequences

## Performance Considerations

- Flash attention is automatically used when running on supported CUDA devices
- Window size can be adjusted based on available memory and sequence length
- Number of key-value heads can be reduced for better efficiency
- For vision models, patch size and number of heads can be tuned based on image size

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{flash-rotary-attention,
  author = {Raktim Mondol},
  title = {PyTorch Flash Rotary Attention},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/raktim-mondol/pytorch-flash-rotary-attention}
}
```

## License

MIT