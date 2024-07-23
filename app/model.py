from tinygrad import nn, Tensor
from .layer import TransformerBlock
import numpy as np
from typing import Optional


class DecisionTransformer:
    def __init__(self, embed_size: int, max_context_length: int, max_timesteps: int, state_dim: int, act_dim: int, n_layers: int = 12, n_heads: int = 12):
        self.embed_size = embed_size
        self.max_content_length = max_context_length

        self.t_embed = Tensor.glorot_uniform(1, max_context_length, embed_size)
        self.embed_global_t = nn.Linear(max_timesteps, embed_size)

        self.embed_s = nn.Linear(state_dim, embed_size)  # convnet
        self.embed_a = nn.Embedding(act_dim, embed_size)
        self.embed_R = nn.Linear(1, embed_size)

        self.layer_norm = nn.LayerNorm(embed_size)

        self.blocks = [TransformerBlock(embed_size, n_heads)
                       for _ in range(n_layers)]

        self.predict_returns = nn.Linear(embed_size, 1)
        self.predict_state = nn.Linear(embed_size, state_dim)
        self.predict_action = nn.Linear(embed_size, act_dim)

    def __call__(self, returns_to_go: float, states: Tensor, actions: Optional[Tensor], timesteps: Tensor):
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, 1, 1)
        batch_size, block_size = states.shape[0], states.shape[1]

        timesteps = timesteps.expand(batch_size, -1, self.embed_size)
        print("t: ", timesteps.shape)

        if actions is not None:
            action_embedding = self.embed_a(actions.squeeze(-1)).tanh()

        returns_to_go_embedding = self.embed_R(returns_to_go).tanh()
        state_embedding = self.embed_s(states).tanh()

        print("s, a, r: ", state_embedding.shape, action_embedding.shape,
              returns_to_go_embedding.shape)

        if actions is not None:
            x = Tensor.stack(
                returns_to_go_embedding,
                state_embedding,
                action_embedding, dim=1).flatten(1, 2)
        else:
            x = Tensor.stack(
                returns_to_go_embedding,
                state_embedding, dim=1).flatten(1, 2)
        print("x stack: ", x.shape)

        pos_e = self.t_embed.repeat(
            1, x.shape[1] // self.max_content_length, 1)
        print("pos_e: ", pos_e.shape)

        embedded_t = timesteps + pos_e

        print("e_t: ", embedded_t.shape)

        x = x + embedded_t
        x = x.dropout()

        mask = self.create_causal_mask(x.shape[1])

        for layer in self.blocks:
            x = layer(x, mask)

        x = self.layer_norm(x)

        print("nroM: ", x.shape)

        returns_preds = self.predict_returns(x[:, 0])
        state_preds = self.predict_state(x[:, 1])
        action_preds = self.predict_action(x[:, 2])

        print("r, s, a: ", returns_preds.shape, state_preds.shape,
              action_preds.shape)

        return returns_preds, state_preds, action_preds

    def create_causal_mask(self, seq_length):
        return Tensor.tril(Tensor.ones((seq_length, seq_length)))

    def sample(self, state, rtgs, actions, timesteps):
        state_cond = state[:, -self.max_context_length // 3:]
        rtgs = rtgs[:, -self.max_context_length // 3:]

        _, _, action_pred = self.forward(
            rtgs, state_cond, actions, timesteps)

        probs = action_pred[-1].softmax()
        next_action = np.random.multinomial(1, probs[-1].numpy())

        return next_action
