import ale_py
import gymnasium as gym
import numpy as np
from tinygrad import TinyJit, Tensor
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save

from .dataset import DQNDataset
from .model import DecisionTransformer


gym.register_envs(ale_py)

# data parameters
dataset_size = 10_000
max_timesteps = 100_000

# model parameters
embed_size = 128
n_heads = 8
n_layers = 6

# training parameters
batch_size = 512
epochs = 5
lr = 6e-4

# game parameters
act_dim = 6
game = "Pong"
max_context_length = 50
state_dim = 84 * 84

def train():
    print("Training RL model...")

    print("Initializing dataset...")
    dataset = DQNDataset(
        state_dim=state_dim, 
        max_context_length=max_context_length, 
        max_timesteps=max_timesteps, 
        size=dataset_size, 
        game=game,
    )

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
    print(f"Parameter count: {np.sum([np.prod(t.shape) for t in parameters]):,}")
    optim = SGD(parameters, lr=lr)

    # Define the training loop
    @TinyJit
    def step(states: Tensor, actions: Tensor, returns_to_go: Tensor, timesteps: Tensor) -> Tensor:
        Tensor.training = True

        targets = actions.squeeze(-1)

        optim.zero_grad()
        out = model(states, actions, returns_to_go, timesteps)

        loss = out.sparse_categorical_crossentropy(targets)
        weighted_loss = loss.mul(returns_to_go).mean()

        weighted_loss.backward()

        optim.step()

        return loss.realize()

    # Train the model
    print("Training model...")
    with Tensor.train():
        for epoch in range(epochs):
            for _step in range(dataset_size // batch_size):
                batch = dataset.sample(batch_size)

                states, actions, returns_to_go, timesteps = batch

                states = Tensor(states)
                actions = Tensor(actions)
                returns_to_go = Tensor(returns_to_go)
                timesteps = Tensor(timesteps)

                loss = step(states, actions, returns_to_go, timesteps)

                print(f"Epoch {epoch+1}, Step {_step+1} | Loss: {loss.numpy()}")

            # Save the model
            state_dict = get_state_dict(model)
            safe_save(state_dict, f"models/model-epoch{epoch+1}.safetensors")
