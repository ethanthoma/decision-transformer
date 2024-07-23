import ale_py
import gymnasium as gym
from tinygrad import Tensor, nn
from tinygrad.nn import optim
import numpy as np

from .model import DecisionTransformer


from .dataset import create_dataset

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

    max_timesteps = 1000

    game = 'Pong'
    data_dir_prefix = '/path/to/dataset/'  # Change this to your dataset path
    num_buffers = 50
    observations, actions, rewards, terminals, next_observations = create_dataset(
        game, data_dir_prefix, num_buffers)
    print(f"Data loaded for {game}. Starting training...")

    model = DecisionTransformer(
        embed_size, max_length, max_length, state_dim, action_dim, n_layers, n_heads)

    optimizer = optim.Adam(nn.state.get_parameters(model), lr=learning_rate)

    trajectories = collect_trajectories(env, model, 100, max_timesteps)

    for epoch in range(num_epochs):
        for batch_idx in range(0, len(trajectories), batch_size):
            batch = trajectories[batch_idx:batch_idx + batch_size]
            states = Tensor(np.array([t["states"] for t in batch]))
            actions = Tensor(np.array([t["actions"] for t in batch]))
            returns_to_go = Tensor(np.array(
                [[sum(t["rewards"][i:]) for i in range(len(t["rewards"]))] for t in batch]))

            print(states.shape, actions.shape, returns_to_go.shape)

            optimizer.zero_grad()

            returns_preds, state_preds, action_preds = model(
                returns_to_go, states, actions, Tensor(np.arange(states.shape[1])))

            loss = ((returns_preds - returns_to_go[:, :-1]) ** 2).mean() + \
                   ((state_preds - states[:, 1:]) ** 2).mean() + \
                nn.CrossEntropyLoss()(action_preds, actions[:, 1:].long())
            loss.backward()
            optimizer.step()

            print(
                f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(trajectories)//batch_size}, Loss: {loss.item()}")

    return model
