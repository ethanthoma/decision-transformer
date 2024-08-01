import ale_py
import gymnasium as gym
import numpy as np
import os
from tinygrad import TinyJit, Tensor
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save

from .dataset import DQNDataset
from .model import DecisionTransformer


gym.register_envs(ale_py)

def train(config: dict):
    print("Training RL model...")

    act_dim = config['act_dim']
    batch_size = config['batch_size']
    data_dir = config['dataset_dir']
    dataset_size = config['dataset_size']
    embed_size = config['embed_size']
    epochs = config['epochs']
    game = config['game']
    lr = config['lr']
    max_context_length = config['max_context_length']
    max_timesteps = config['max_timesteps']
    model_dir = config['model_dir']
    n_heads = config['n_heads']
    n_layers = config['n_layers']
    num_checkpoints = config['num_checkpoints']
    split = config['split']
    state_dim = config['state_dim']
    train_set_path = config['train_set_path']

    print("Initializing dataset...")
    dataset = DQNDataset(
        state_dim=state_dim, 
        max_context_length=max_context_length, 
        max_timesteps=max_timesteps, 
        game=game,
        data_dir=data_dir,
        size=dataset_size, 
        split=split, 
        num_checkpoints=num_checkpoints, 
        train_set_path=train_set_path,
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
            safe_save(state_dict, os.path.join(model_dir, f"model-epoch{epoch+1}.safetensors"))
