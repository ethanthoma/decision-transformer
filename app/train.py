import ale_py
import gymnasium as gym
import numpy as np
from tinygrad import Tensor
from tinygrad.nn.state import get_parameters
from tinygrad.nn.optim import SGD
from typing import Tuple

from tinygrad import TinyJit
from tinygrad.nn.state import safe_save, get_state_dict

from .model import DecisionTransformer
from .buffer import ReplayBuffer


gym.register_envs(ale_py)

# data parameters
max_timesteps = 100_000
dataset_size = 10_000

# model parameters
embed_size = 128
n_layers = 6
n_heads = 8

# training parameters
batch_size = 128
epochs = 5
lr = 6e-4

# game parameters
game_name = "Pong-v4"
max_context_length = 50
state_dim = 210 * 160
act_dim = 6

def train_rl():
    print("Training RL model...")

    # Initialize the environment
    print("Initializing environment...")
    env = gym.make(game_name, obs_type="grayscale")

    print("Initializing replay buffer...")
    buffer = makeBuffer(dataset_size, state_dim, max_context_length, max_timesteps, env)

    # Initialize the model
    print("Initializing model...")
    model = DecisionTransformer(
        embed_size=embed_size,
        max_context_length=max_context_length,
        state_dim=state_dim,
        act_dim=act_dim,
        n_layers=n_layers,
        n_heads=n_heads,
    )

    # Define the optimizer and loss function
    parameters = get_parameters(model)
    print(f"parameter count {np.sum([np.prod(t.shape) for t in parameters]):,}")
    optim = SGD(parameters, lr=lr)

    @TinyJit
    def jit(*args):
        return model(*args).realize()

    # Define the training loop
    def step(states: Tensor, actions: Tensor, returns_to_go: Tensor, timesteps: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        Tensor.training = True

        optim.zero_grad()
        out = jit(states, actions, returns_to_go, timesteps)

        loss = out.sparse_categorical_crossentropy(targets.squeeze(-1)).mul(returns_to_go).mean().backward()
        optim.step()

        cat = out.argmax(axis=-1).unsqueeze(-1)
        accuracy = (cat == targets).mean()

        return loss.realize(), accuracy.realize()

    # Train the model
    print("Training model...")
    with Tensor.train():
        for epoch in range(epochs):
            for _step in range(dataset_size // batch_size):
                batch = buffer.sample(batch_size)

                states, actions, returns_to_go, timesteps = batch

                states = Tensor(states, requires_grad=True)
                actions = Tensor(actions)
                returns_to_go = Tensor(returns_to_go, requires_grad=True)
                timesteps = Tensor(timesteps, requires_grad=True)

                targets = actions

                loss, accuracy = step(states, actions, returns_to_go, timesteps, targets)

                print(f"Epoch {epoch+1}, Step {_step+1} | Loss: {loss.numpy()}, Accuracy: {accuracy.numpy()}")

    state_dict = get_state_dict(model)
    safe_save(state_dict, "model.safetensors")


def makeBuffer(dataset_size: int, state_dim: int, max_context_length: int, max_timesteps: int, env):
    buffer = ReplayBuffer(dataset_size, state_dim, max_context_length)
    obs = None
    done = None
    t = 0
    rtg = 0

    states = np.zeros([max_timesteps, state_dim])
    actions = np.zeros([max_timesteps, 1])
    rewards = np.zeros([max_timesteps, 1])
    timesteps = np.zeros([max_timesteps, 1])

    for _ in range(dataset_size):
        if t == 0:
            obs = env.reset()[0]

        t += 1

        action = env.action_space.sample()
        next_obs, reward, done, _, _ = env.step(action)
        rtg += reward

        states[t, :] = np.array(obs).flatten()
        actions[t, :] = action
        rewards[t, :] = reward
        timesteps[t, :] = t

        obs = next_obs

        if done or t == max_timesteps:
            done_idx = t

            offset = np.random.randint(0, done_idx)
            length = min(done_idx-offset, max_context_length)
            end = offset + length

            rewards = rewards.max() - rewards

            buffer.add(
                states=states[offset:end], 
                actions=actions[offset:end], 
                returns_to_go=rewards[offset:end], 
                timesteps=timesteps[offset:end],
                done_idx=done_idx,
            )

            done = False
            t = 0
            rtg = 0

    return buffer
