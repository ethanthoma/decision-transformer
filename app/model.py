from tinygrad import nn, Tensor
from .layer import TransformerBlock
import numpy as np


class DecisionTransformer:
    def __init__(self, embed_size: int, max_context_length: int, state_dim: int, act_dim: int, n_layers: int = 12, n_heads: int = 12):
        self.embed_size = embed_size
        self.max_context_length = max_context_length

        self.embed_t = nn.Linear(embed_size, embed_size)

        self.embed_s = nn.Linear(state_dim, embed_size)
        self.embed_a = nn.Embedding(act_dim, embed_size)
        self.embed_R = nn.Linear(1, embed_size)

        self.blocks = [TransformerBlock(embed_size, n_heads)
                       for _ in range(n_layers)]

        self.layer_norm = nn.LayerNorm(embed_size)

        self.head = nn.Linear(embed_size, act_dim, bias=False)

    def __call__(self, states: Tensor, actions: Tensor, returns_to_go: Tensor, timesteps: Tensor) -> Tensor:
        # states: (batch, block_size, 210 * 160)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, 1, 1)
        batch_size = states.shape[0]
        step_size = 3

        timesteps = timesteps.expand(batch_size, -1, self.embed_size)
        timesteps_embedding = self.embed_t(timesteps)
        position_embedding = timesteps_embedding.repeat(1, step_size, 1)

        state_embedding = self.embed_s(states).tanh()
        action_embedding = self.embed_a(actions.squeeze(-1)).tanh()
        returns_to_go_embedding = self.embed_R(returns_to_go).tanh()

        x = Tensor.stack(
            returns_to_go_embedding.unsqueeze(-1),
            state_embedding.unsqueeze(-1),
            action_embedding.unsqueeze(-1), 
            dim=-1
        ).reshape(batch_size, -1, self.embed_size)

        x = x + position_embedding
        x = x.dropout()

        mask = self.create_causal_mask(x.shape[1])

        for layer in self.blocks:
            x = layer(x, mask)

        x = self.layer_norm(x)

        x = x[:, ::step_size, :]

        x = self.head(x)

        return x

    def create_causal_mask(self, seq_length: int) -> Tensor:
        return Tensor.tril(Tensor.ones((seq_length, seq_length)))

    def sample(self, state: Tensor, rtgs: Tensor, actions: Tensor, timesteps: Tensor) -> Tensor:
        state_cond = state[:, -self.max_context_length // 3:]
        rtgs = rtgs[:, -self.max_context_length // 3:]

        _, _, action_pred = self(state_cond, actions, rtgs, timesteps)

        probs = action_pred[-1].softmax()
        next_action = np.random.multinomial(1, probs[-1].numpy())

        return Tensor(next_action)
