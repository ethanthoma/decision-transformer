import numpy as np
import os
from tinygrad import TinyJit, Tensor, dtypes
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save

from .dataset import DQNDataset
from .model import DecisionTransformer
from .optimizer import DT_Optimizer
from .scheduler import DT_Scheduler


def train(config: dict):
    print("Training RL model...")

    act_dim = config["act_dim"]
    batch_size = config["batch_size"]
    beta_1 = config["beta_1"]
    beta_2 = config["beta_2"]
    data_dir = config["dataset_dir"]
    dataset_size = config["dataset_size"]
    embed_size = config["embed_size"]
    epochs = config["epochs"]
    final_tokens = config["final_tokens"]
    game = config["game"]
    loop = config["loop"]
    lr = config["lr"]
    max_concurrent = config["max_concurrent"]
    max_context_length = config["max_context_length"]
    max_timesteps = config["max_timesteps"]
    model_dir = config["model_dir"]
    n_heads = config["n_heads"]
    n_layers = config["n_layers"]
    num_checkpoints = config["num_checkpoints"]
    split = config["split"]
    state_dim = config["state_dim"]
    save_dir = config["save_dir"]
    warmup_tokens = config["warmup_tokens"]
    weight_decay = config["weight_decay"]

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
        save_dir=save_dir,
        max_concurrent=max_concurrent,
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
        loop=loop,
    )

    # Define the optimizer and loss function
    parameters = get_parameters(model)
    print(f"Parameter count: {np.sum([np.prod(t.shape) for t in parameters]):,}")

    optim = DT_Optimizer(model, lr=lr, b1=beta_1, b2=beta_2, weight_decay=weight_decay)
    scheduler = DT_Scheduler(
        optim, lr=lr, warmup_tokens=warmup_tokens, final_tokens=final_tokens
    )

    # Define the training loop
    @TinyJit
    def step(
        states: Tensor,
        actions: Tensor,
        returns_to_go: Tensor,
        timesteps: Tensor,
        tokens: int = 0,
    ) -> Tensor:
        Tensor.training = True

        targets = actions.squeeze(-1)

        optim.zero_grad()
        out = model(states, actions, returns_to_go, timesteps)

        loss = out.sparse_categorical_crossentropy(targets).mean()

        loss.backward()

        optim.step()
        scheduler.step(tokens)

        return loss.realize()

    def data_generator(batch_size):
        while True:
            yield dataset.sample(batch_size)

    # Train the model
    print("Training model...")
    with Tensor.train():
        data_gen = data_generator(batch_size)
        tokens: int = 0
        for epoch in range(epochs):
            for _step in range(dataset_size // batch_size):
                batch = next(data_gen)

                states, actions, returns_to_go, timesteps = batch

                states = Tensor(states, dtype=dtypes.uint8)
                actions = Tensor(actions, dtype=dtypes.uint8)
                returns_to_go = Tensor(returns_to_go, dtype=dtypes.bfloat16)
                timesteps = Tensor(timesteps, dtype=dtypes.uint8)

                tokens += int(actions.sign().sum().numpy())

                loss = step(states, actions, returns_to_go, timesteps, tokens)

                print(f"Epoch {epoch+1}, Step {_step+1} | Loss: {loss.numpy()}")

            # Save the model
            state_dict = get_state_dict(model)
            safe_save(
                state_dict, os.path.join(model_dir, f"model-epoch{epoch+1}.safetensors")
            )
