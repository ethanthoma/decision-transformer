from tinygrad import nn, Tensor
from .layer import *


class GPT:
    def __init__(self, embed_size: int, n_layers: int, vocab_size: int, max_seq_len: int, n_heads: int = 8):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.positional_embedding = nn.Embedding(max_seq_len, embed_size)
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = Tensor.dropout
        self.blocks = [TransformerBlock(embed_size, n_heads)
                       for _ in range(n_layers)]
        self.layer_norm = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size, bias=False)

    def __call__(self, x: Tensor):
        seq_len = x.shape[1]
        assert seq_len <= self.max_seq_len
        pos = Tensor.arange(seq_len)

        token_embedding = self.token_embedding(x)
        positional_embedding = self.positional_embedding(pos)
        print(token_embedding.shape, positional_embedding.shape)

        x = self.dropout(token_embedding + positional_embedding)

        mask = Tensor.fill((seq_len, seq_len), -1e9)
        for block in self.blocks:
            x = block(x, mask)

        x = self.layer_norm(x)
        x = self.head(x)

        return x


class DecisionTransformer:
    def __init__(self, embed_size: int, max_length: int, state_dim: int, action_dim: int, n_layers: int = 12, n_heads: int = 12):
        self.embed_size = embed_size

        self.timestep_embedding = nn.Embedding(max_length, embed_size)
        self.returns_to_go_embedding = nn.Embedding(1, embed_size)
        self.state_embedding = nn.Embedding(state_dim, embed_size)
        self.action_embedding = nn.Embedding(action_dim, embed_size)

        self.layer_norm = nn.LayerNorm(embed_size)

        self.blocks = [TransformerBlock(embed_size, n_heads)
                       for _ in range(n_layers)]

        self.predict_returns = nn.Linear(embed_size, 1)
        self.predict_state = nn.Linear(embed_size, state_dim)
        self.predict_action = nn.Linear(embed_size, action_dim)

    def __call__(self, returns_to_go: float, states: Tensor, actions: Tensor, timesteps: Tensor):
        batch_size, seq_length = states.shape[0], states.shape[1]

        timestep_embedding = self.positional_embedding(timesteps)

        state_embedding = self.state_embedding(states) + timestep_embedding
        action_embedding = self.action_embedding(
            actions) + timestep_embedding
        returns_to_go_embedding = self.returns_to_go_embedding(
            returns_to_go) + timestep_embedding

        x = Tensor.stack([
            returns_to_go_embedding,
            state_embedding,
            action_embedding], axis=1).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.embed_size)
        x = self.layer_norm(x)

        mask = Tensor.ones(batch_size, seq_length)
        mask = Tensor.stack((mask, mask, mask), dim=1).permute(
            0, 2, 1).reshape(batch_size, 3*seq_length)

        for layer in self.blocks:
            x = layer(x, mask)

        x = x.reshape(batch_size, seq_length, 3,
                      self.hidden_size).permute(0, 2, 1, 3)

        returns_preds = self.predict_returns(x[:, 0])
        state_preds = self.predict_state(x[:, 1])
        action_preds = self.predict_action(x[:, 2])

        return returns_preds, state_preds, action_preds
