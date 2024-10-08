import numpy as np
from tinygrad import nn, Tensor
from typing import Optional


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
        query, key, value = Tensor.einsum("bij,jkt->tbik", x, self.qkv)

        query = Tensor.einsum("bij,hjk->bhik", query, self.w_q)
        key = Tensor.einsum("bij,hjk->bhik", key, self.w_k)
        value = Tensor.einsum("bij,hjk->bhik", value, self.w_v)

        attention = Tensor.scaled_dot_product_attention(query, key, value, mask)

        attention = attention.dropout(0.6)

        attention = attention.permute(0, 2, 1, 3)
        attention = attention.flatten(start_dim=2)
        attention = self.fc_out(attention)

        attention = attention.dropout(0.6)

        return attention


class TransformerBlock:
    def __init__(self, embed_size: int, n_heads: int, mask: Optional[Tensor]):
        self.mask = mask
        self.attention = MultiHeadSelfAttention(embed_size, n_heads)
        self.feed_forward = [
            nn.Linear(embed_size, 4 * embed_size),
            Tensor.gelu,
            nn.Linear(4 * embed_size, embed_size),
            Tensor.dropout,
        ]
        self.norm_one = nn.LayerNorm(embed_size)
        self.norm_two = nn.LayerNorm(embed_size)

    def __call__(self, x: Tensor, mask: Optional[Tensor]):
        h = self.norm_one(x)
        x = x + self.attention(h, mask if mask is not None else self.mask)
        h = self.norm_two(x)
        x = x + h.sequential(self.feed_forward)
        return x


class LoopTransformerBlock:
    def __init__(
        self, embed_size: int, n_heads: int, mask: Optional[Tensor], loops: int
    ):
        self.mask = mask
        self.attention = MultiHeadSelfAttention(embed_size, n_heads)
        self.feed_forward = [
            nn.Linear(embed_size, 4 * embed_size),
            Tensor.gelu,
            nn.Linear(4 * embed_size, embed_size),
            Tensor.dropout,
        ]
        self.norm_one = nn.LayerNorm(embed_size)
        self.norm_two = nn.LayerNorm(embed_size)
        self.loops = loops

    def __call__(self, x: Tensor, mask: Optional[Tensor]):
        for _ in range(self.loops):
            h = self.norm_one(x)
            x = x + self.attention(h, mask if mask is not None else self.mask)
            h = self.norm_two(x)
            x = x + h.sequential(self.feed_forward)
        return x


class StateEncoder:
    def __init__(self, embed_size: int):
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.linear = nn.Linear(3136, embed_size)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).relu().flatten(start_dim=1)
        x = self.linear(x).tanh()
        return x


class DecisionTransformer:
    def __init__(
        self,
        embed_size: int,
        context_length: int,
        state_dim: int,
        act_dim: int,
        n_layers: int,
        n_heads: int,
        max_timesteps: int,
        loop: bool = False,
    ):
        self.embed_size = embed_size
        self.context_length = context_length

        self.embed_t = nn.Embedding(max_timesteps, embed_size)

        self.embed_s = StateEncoder(embed_size)
        self.embed_a = nn.Embedding(act_dim, embed_size)
        self.embed_R = nn.Linear(1, embed_size)

        if loop:
            self.blocks = [
                LoopTransformerBlock(embed_size, n_heads, None, loops=n_layers)
            ]
        else:
            self.blocks = [
                TransformerBlock(embed_size, n_heads, None) for _ in range(n_layers)
            ]

        self.layer_norm = nn.LayerNorm(embed_size)

        self.head = nn.Linear(embed_size, act_dim, bias=False)

        parameters = nn.state.get_parameters(self)
        print(f"Parameter count: {np.sum([np.prod(t.shape) for t in parameters]):,}")

    def __call__(
        self,
        states: Tensor,
        actions: Tensor,
        returns_to_go: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        # states: (batch, context_length, 4 * 84 * 84)
        # actions: (batch, context_length - 1, 1)
        # rtgs: (batch, context_length, 1)
        # timesteps: (batch, context_length, 1)
        batch_size, context_length = states.shape[:2]
        step_size = 3

        timesteps_embedding = self.embed_t(timesteps.squeeze(-1))

        states = states.reshape(-1, 4, 84, 84)
        state_embedding = self.embed_s(states).reshape(batch_size, -1, self.embed_size)
        state_embedding = state_embedding + timesteps_embedding

        action_embedding = self.embed_a(actions.squeeze(-1)).tanh()
        if action_embedding.shape[1] < context_length:
            action_embedding = action_embedding.pad((None, (0, 1), None))
        action_embedding = action_embedding + timesteps_embedding

        returns_to_go_embedding = self.embed_R(returns_to_go).tanh()
        returns_to_go_embedding = returns_to_go_embedding + timesteps_embedding

        x = Tensor.stack(
            returns_to_go_embedding.unsqueeze(-1),
            state_embedding.unsqueeze(-1),
            action_embedding.unsqueeze(-1),
            dim=-1,
        ).reshape(batch_size, -1, self.embed_size)

        x = x.dropout()

        mask = Tensor.full(
            (1, 1, context_length * step_size, context_length * step_size),
            float("-inf"),
            dtype=x.dtype,
        ).triu(1)

        for block in self.blocks:
            x = block(x, mask)

        x = self.layer_norm(x)

        x = x[:, 1::step_size, :]  # only take state embeddings

        x = self.head(x)

        return x
