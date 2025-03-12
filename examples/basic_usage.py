import torch
import torch.nn as nn
from attention import FlashAttention, SlidingWindowAttention

def test_flash_attention():
    """
    Demonstrates basic usage of FlashAttention
    """
    print("\n=== Testing Flash Attention ===")
    
    # Create a sample input
    batch_size = 2
    seq_length = 256
    dim = 512
    x = torch.randn(batch_size, seq_length, dim)
    
    # Initialize attention module
    attn = FlashAttention(
        dim=dim,
        heads=8,
        dim_head=64,
        dropout=0.1
    )
    
    # Process input
    out = attn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Self-attention check passed: {out.shape == x.shape}")

def test_sliding_window_attention():
    """
    Demonstrates usage of SlidingWindowAttention with long sequences
    """
    print("\n=== Testing Sliding Window Attention ===")
    
    # Create a long sequence input
    batch_size = 1
    seq_length = 2048  # Long sequence
    dim = 512
    window_size = 512
    x = torch.randn(batch_size, seq_length, dim)
    
    # Initialize sliding window attention
    attn = SlidingWindowAttention(
        dim=dim,
        heads=8,
        dim_head=64,
        window_size=window_size,
        dropout=0.1
    )
    
    # Process input
    out = attn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Long sequence check passed: {out.shape == x.shape}")

class SimpleTransformerBlock(nn.Module):
    """
    Example of using attention in a transformer block
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        self.attention = FlashAttention(dim, heads, dim_head, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Attention block
        x = x + self.dropout(self.attention(self.norm1(x)))
        # FFN block
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

def test_transformer_block():
    """
    Demonstrates using attention within a transformer block
    """
    print("\n=== Testing Transformer Block ===")
    
    # Create sample input
    batch_size = 4
    seq_length = 128
    dim = 512
    x = torch.randn(batch_size, seq_length, dim)
    
    # Initialize transformer block
    block = SimpleTransformerBlock(dim=dim)
    
    # Process input
    out = block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Transformer block check passed: {out.shape == x.shape}")

class TextClassifier(nn.Module):
    """
    Example of using attention for text classification
    """
    def __init__(self, vocab_size, dim, num_classes, max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.attention = SlidingWindowAttention(
            dim=dim,
            window_size=256,
            heads=8,
            dim_head=64,
            dropout=0.1
        )
        self.norm = nn.LayerNorm(dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, dim]
        x = self.attention(x)  # Apply attention
        x = self.norm(x)
        x = x.transpose(1, 2)  # [batch_size, dim, seq_len]
        x = self.pool(x).squeeze(-1)  # [batch_size, dim]
        return self.fc(x)  # [batch_size, num_classes]

def test_text_classifier():
    """
    Demonstrates using attention for text classification
    """
    print("\n=== Testing Text Classifier ===")
    
    # Create sample input
    batch_size = 8
    seq_length = 256
    vocab_size = 10000
    dim = 512
    num_classes = 5
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Initialize classifier
    classifier = TextClassifier(
        vocab_size=vocab_size,
        dim=dim,
        num_classes=num_classes
    )
    
    # Process input
    out = classifier(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Classification output check passed: {out.shape == (batch_size, num_classes)}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run all tests
    test_flash_attention()
    test_sliding_window_attention()
    test_transformer_block()
    test_text_classifier()