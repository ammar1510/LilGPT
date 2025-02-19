import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer which first projects inputs into Q, K, V
    then applies scaled dot-product attention in parallel across the heads.
    """
    def __init__(self, in_dim: int, n_heads: int, dropout: float = 0.2) -> None:
        super().__init__()
        assert in_dim % n_heads == 0, "in_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = in_dim // n_heads

        # Single projection layer to obtain Q, K, and V
        self.qkv_proj = nn.Linear(in_dim, 3 * in_dim)
        self.out_proj = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, in_dim)
        Returns:
            Tensor of shape (B, T, in_dim)
        """
        B, T, _ = x.shape
        qkv = self.qkv_proj(x)  # (B, T, 3*in_dim)
        # Reshape into (B, T, 3, n_heads, head_dim) then rearrange to (3, B, n_heads, T, head_dim)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, n_heads, T, head_dim)

        # Compute attention using the built-in function with causal masking
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p, is_causal=True)
        # Merge attention head outputs: (B, n_heads, T, head_dim) -> (B, T, in_dim)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, -1).contiguous()
        return self.out_proj(attn_out)


class FeedForward(nn.Module):
    """
    Feed-forward network: a simple 2-layer MLP with ReLU and dropout.
    """
    def __init__(self, in_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 4 * in_dim),
            nn.ReLU(),
            nn.Linear(4 * in_dim, in_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerLayer(nn.Module):
    """
    Transformer layer block with multi-head attention and feed-forward network,
    each with its own LayerNorm and residual connection.
    """
    def __init__(self, in_dim: int, n_heads: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(in_dim, n_heads, dropout=dropout)
        self.ffn = FeedForward(in_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block with residual connection.
        x = x + self.attn(self.norm1(x))
        # Feed-forward block with residual connection.
        x = x + self.ffn(self.norm2(x))
        return self.dropout(x)


class GPT(nn.Module):
    """
    A GPT-style transformer model.
    
    Attributes:
        tok_emb: token embedding layer.
        pos_emb: positional embedding layer.
        layers: sequential transformer blocks.
        out: final projection layer mapping to vocabulary logits.
        criterion: cross-entropy loss used when targets are provided.
    """
    def __init__(self, vocab_size: int, context_len: int, n_layers: int, in_dim: int, n_heads: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, in_dim)
        self.pos_emb = nn.Embedding(context_len, in_dim)
        self.context_len = context_len

        self.layers = nn.Sequential(*[TransformerLayer(in_dim, n_heads, dropout) for _ in range(n_layers)])
        self.out = nn.Linear(in_dim, vocab_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the GPT model.
        
        Args:
            x: Input tensor of shape (B, T) containing token ids.
            target: Optional tensor of shape (B, T) for computing the loss.
            
        Returns:
            logits: Output tensor of shape (B, T, vocab_size).
            loss: Cross-entropy loss if target is provided, otherwise None.
        """
        B, T = x.shape
        # Generate positions and embed (broadcasting over batch dimension)
        pos = torch.arange(T, device=x.device)
        x = self.tok_emb(x) + self.pos_emb(pos)  # (B, T, in_dim)
        x = self.layers(x)
        logits = self.out(x)  # (B, T, vocab_size)

        loss = None
        if target is not None:
            logits_flat = logits.view(-1, logits.size(-1))
            target_flat = target.view(-1)
            loss = self.criterion(logits_flat, target_flat)
        return logits, loss

    def generate(self, input: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """
        Autoregressively generate tokens given an input sequence.
        
        Args:
            input: Tensor of shape (B, T) containing input token ids.
            num_tokens: Number of tokens to generate.
            
        Returns:
            Tensor of shape (B, T + num_tokens) with the generated sequence.
        """
        self.eval()
        with torch.no_grad():
            for _ in range(num_tokens):
                # Use the last context tokens for generation.
                cropped_input = input[:, -self.context_len :]
                logits, _ = self.forward(cropped_input)
                # Get logits for the last time step and compute probabilities.
                logits_last = logits[:, -1, :]
                probs = F.softmax(logits_last, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input = torch.cat([input, next_token], dim=1)
        return input

    @classmethod
    def create(
        cls,
        vocab_size: int,
        context_len: int,
        n_layers: int,
        in_dim: int,
        n_heads: int,
        dropout: float = 0.2,
        compile_model: bool = False
    ) -> "GPT":
        """
        Factory method to create a GPT model. If compile_model is True and you're using PyTorch 2.0+,
        the model will be compiled with torch.compile for potential speed improvements.
        """
        model = cls(vocab_size, context_len, n_layers, in_dim, n_heads, dropout)
        if compile_model:
            model = torch.compile(model)
        return model




