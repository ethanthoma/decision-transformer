from typing import Optional
from tinygrad import Tensor, nn


class MultiHeadSelfAttention:
    def __init__(self, embed_size: int, n_heads: int):
        self.embed_size = embed_size
        self.n_heads = n_heads

        self.head_dim = embed_size // n_heads

        assert self.head_dim * n_heads == embed_size

        self.qkv = Tensor.randn(embed_size, embed_size, 3)

        self.w_q = Tensor.randn(n_heads, embed_size, self.head_dim)
        self.w_k = Tensor.randn(n_heads, embed_size, self.head_dim)
        self.w_v = Tensor.randn(n_heads, embed_size, self.head_dim)

        self.fc_out = nn.Linear(embed_size, embed_size, bias=False)

    def __call__(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # x shape: (n, seq_len, embed_size)
        # mask shape: (n, seq_len, seq_len)
        query, key, value = Tensor.einsum('bij,jkt->tbik', x, self.qkv)

        query = Tensor.einsum('bij,hjk->bhik', query, self.w_q)
        key = Tensor.einsum('bij,hjk->bhik', key, self.w_k)
        value = Tensor.einsum('bij,hjk->bhik', value, self.w_v)

        attention = Tensor.scaled_dot_product_attention(
            query, key, value, mask)

        attention = attention.dropout(0.6)

        attention = attention.permute(0, 2, 1, 3)
        attention = attention.flatten(start_dim=2)
        attention = self.fc_out(attention)

        attention = attention.dropout(0.6)

        return attention


class TransformerBlock:
    def __init__(self, embed_size: int, n_heads: int):
        self.attention = MultiHeadSelfAttention(embed_size, n_heads)
        self.feed_forward = [
            nn.Linear(embed_size, 4 * embed_size),
            nn.Linear(4 * embed_size, embed_size),
            Tensor.gelu,
            Tensor.dropout,
        ]
        self.norm_one = nn.LayerNorm(embed_size)
        self.norm_two = nn.LayerNorm(embed_size)

    def __call__(self, x: Tensor, mask: Optional[Tensor] = None):
        h = self.norm_one(x)
        h = x + self.attention(h, mask)
        h = self.norm_two(h)
        h = h + h.sequential(self.feed_forward)
        return h
