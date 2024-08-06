import ale_py
from collections import deque
import gymnasium as gym
import numpy as np
from numpy._typing import NDArray
import os
from tinygrad import Tensor, TinyJit
from tinygrad.dtype import dtypes
from tinygrad.nn.state import safe_load, load_state_dict
from typing import Tuple

from .model import DecisionTransformer

gym.register_envs(ale_py)


@TinyJit
def interpolate(x: NDArray, size: Tuple[int, ...] = (84, 84)):
    tensor: Tensor = Tensor(x)
    expand = list(x.shape)

    for i in range(-len(size), 0):
        scale = tensor.shape[i] / size[i]
        arr, reshape = Tensor.arange(size[i], dtype=dtypes.float32), [1] * tensor.ndim
        index = (scale * (arr + 0.5) - 0.5).clip(0, tensor.shape[i] - 1)
        reshape[i] = expand[i] = size[i]
        low, high, perc = [
            y.reshape(reshape).expand(expand)
            for y in (index.floor(), index.ceil(), index - index.floor())
        ]
        tensor = tensor.gather(i, low).lerp(tensor.gather(i, high), perc)

    return tensor.numpy()


def process_frame(state: NDArray, frames: deque) -> Tuple[NDArray, deque]:
    state = interpolate(state)
    frames.append(state)
    state = np.stack(frames).flatten()
    return state, frames


def test(config: dict):
    act_dim = config["act_dim"]
    embed_size = config["embed_size"]
    game = config["game"]
    game_version = config["game_version"]
    n_heads = config["n_heads"]
    n_layers = config["n_layers"]
    max_context_length = config["max_context_length"]
    model_dir = config["model_dir"]
    model_name = config["model_name"]
    num_episodes = config["num_episodes"]
    render_mode = config["render_mode"]
    state_dim = config["state_dim"]
    target_return = config["target_return"]

    # Load the environment
    game_name = f"{game}-{game_version}"
    env = gym.make(game_name, obs_type="grayscale", render_mode=render_mode)

    # Load the model
    model = DecisionTransformer(
        embed_size=embed_size,
        max_context_length=max_context_length,
        state_dim=state_dim,
        act_dim=act_dim,
        n_layers=n_layers,
        n_heads=n_heads,
    )

    # Load the saved weights
    state_dict = safe_load(os.path.join(model_dir, model_name))
    load_state_dict(model, state_dict)

    total_rewards = []
    Tensor.training = False
    for episode in range(num_episodes):
        state = env.reset()[0]
        state = interpolate(state)
        frames = deque([state] * 4, maxlen=4)
        state, frames = process_frame(state, frames)

        done = False
        episode_reward = 0

        R = [target_return]
        s = [state]
        a = []
        t = [1]

        while not done:
            states_tensor = Tensor(s).unsqueeze(0)
            actions_tensor = Tensor(a).unsqueeze(0).unsqueeze(-1)
            returns_to_go_tensor = Tensor(np.array(R)).unsqueeze(0).unsqueeze(-1)
            timesteps_tensor = Tensor(np.array(t)).unsqueeze(0).unsqueeze(-1)

            # sample next action
            action_preds = model(
                states_tensor, actions_tensor, returns_to_go_tensor, timesteps_tensor
            )
            action = action_preds[0, -1].argmax().item()

            state, reward, done, _, _ = env.step(action)
            episode_reward += reward

            # append new tokens to sequence
            state, frames = process_frame(state, frames)
            R.append(R[-1] - reward)  # decrement returns-to-go with reward
            s.append(state)
            a.append(action)
            t.append(len(R))

            # only keep context length of K
            R = R[-max_context_length:]
            s = s[-max_context_length:]
            a = a[-max_context_length + 1 :]
            t = t[-max_context_length:]

        total_rewards.append(episode_reward)
        print(
            f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}, Steps: {t[-1]}"
        )

    avg_reward = np.mean(total_rewards)
    print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward}")

    env.close()
