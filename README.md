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

## Usage

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