import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class RotaryEmbedding(nn.Module):
    """
    Implements Rotary Position Embedding (RoPE) as described in the paper
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, freqs):
    """Apply rotary positional embeddings to queries and keys."""
    q_rot = q * freqs.cos() + rotate_half(q) * freqs.sin()
    k_rot = k * freqs.cos() + rotate_half(k) * freqs.sin()
    return q_rot, k_rot

class FlashAttention(nn.Module):
    """
    Implements Flash Attention with Rotary Embeddings and Grouped Query Attention.
    Uses PyTorch's native flash attention when available.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Rotary embeddings
        self.rotary = RotaryEmbedding(dim_head)
        
        # Grouped query attention
        self.num_kv_heads = heads  # Start with same number of heads
        self.kv_proj = nn.Linear(dim, dim_head * self.num_kv_heads * 2, bias=False)

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, dim]
            mask (torch.Tensor, optional): Attention mask
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, dim]
        """
        b, n, _, h = *x.shape, self.heads
        
        # Project queries, keys, values
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        # Apply rotary embeddings
        freqs = self.rotary(n, x.device)
        q, k = apply_rotary_pos_emb(q, k, freqs)
        
        # Grouped query attention
        k = k.repeat_interleave(h // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(h // self.num_kv_heads, dim=1)
        
        # Flash attention implementation
        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.
            )
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class SlidingWindowAttention(FlashAttention):
    """
    Extends FlashAttention with a sliding window mechanism for handling long sequences.
    Processes the input sequence in overlapping windows for efficiency.
    """
    def __init__(self, dim, window_size=512, **kwargs):
        super().__init__(dim, **kwargs)
        self.window_size = window_size

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, dim]
            mask (torch.Tensor, optional): Attention mask
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, dim]
        """
        b, n, _ = x.shape
        if n <= self.window_size:
            return super().forward(x, mask)
            
        # Split into windows
        windows = x.unfold(1, self.window_size, self.window_size // 2)
        windows = windows.contiguous().view(b * windows.size(1), self.window_size, -1)
        
        # Process each window
        out = super().forward(windows)
        
        # Reconstruct sequence
        out = out.view(b, -1, out.size(-1))
        return out[:, :n, :]  # Trim padding

if __name__ == "__main__":
    # Example usage and test
    attn = SlidingWindowAttention(dim=512, heads=8, dim_head=64, window_size=512)
    x = torch.randn(1, 1024, 512)  # [batch_size, seq_len, dim]
    out = attn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "Output shape should match input shape"