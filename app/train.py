import ale_py
import gymnasium as gym
from tinygrad import Tensor

from .model import DecisionTransformer

gym.register_envs(ale_py)

env = gym.make("ALE/Pong-v5")


def train_decision_transformer():
    # pong
    n_layers: int = 6
    n_heads: int = 8
    embed_size: int = 128
    max_length: int = 50
    return_to_go_conditioning: int = 20

    state_dim: int = 210 * 160 * 3
    action_dim: int = 18
    learning_rate: float = 6e-4
    batch_size: int = 512
    num_epochs: int = 5

    model = DecisionTransformer(
        embed_size, max_length, state_dim, action_dim, n_layers, n_heads)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
