import math
import os
from tinygrad import TinyJit, Tensor, dtypes
from tinygrad.helpers import tqdm
from tinygrad.nn.optim import AdamW, OptimizerGroup
from tinygrad.nn.state import get_state_dict, safe_save, get_parameters
import gc

from .dataset import DQNDataset
from .model import DecisionTransformer
from .scheduler import DT_Scheduler, LRSchedulerGroup


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

    # ** Dataset **
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

    # ** Model **
    print("Initializing model...")
    model = DecisionTransformer(
        embed_size=embed_size,
        context_length=max_context_length,
        state_dim=state_dim,
        act_dim=act_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_timesteps=max_timesteps,
        loop=loop,
    )
    parameters = get_parameters(model)

    # Setup the optimizer and scheduler
    # optim = DT_Optimizer(model, lr=lr, b1=beta_1, b2=beta_2, weight_decay=weight_decay)
    # scheduler = DT_Scheduler( optim, lr=lr, warmup_tokens=warmup_tokens, final_tokens=final_tokens)

    # ** Optimizer **
    weight_decay_list = [
        v
        for k, v in get_state_dict(model).items()
        if "norm" in k or "bias" in k or "embed_a" in k
    ]
    no_weight_decay_list = [p for p in parameters if p not in set(weight_decay_list)]

    optim_weight_decay = AdamW(
        weight_decay_list, lr=lr, b1=beta_1, b2=beta_2, weight_decay=weight_decay
    )
    optim_no_weight_decay = AdamW(
        no_weight_decay_list, lr=lr, b1=beta_1, b2=beta_2, weight_decay=0.0
    )
    optim_group = OptimizerGroup(optim_weight_decay, optim_no_weight_decay)

    # ** LR Scheduler **
    scheduler_weight_decay = DT_Scheduler(
        optim_weight_decay,
        lr=lr,
        warmup_tokens=warmup_tokens,
        final_tokens=final_tokens,
    )
    scheduler_no_weight_decay = DT_Scheduler(
        optim_no_weight_decay,
        lr=lr,
        warmup_tokens=warmup_tokens,
        final_tokens=final_tokens,
    )
    scheduler_group = LRSchedulerGroup(
        scheduler_weight_decay, scheduler_no_weight_decay
    )

    # ** Loss Function **
    def lossfn(out, y):
        return out.sparse_categorical_crossentropy(y.squeeze(-1)).mean()

    # ** Training Step **
    @TinyJit
    def step(
        states: Tensor,
        actions: Tensor,
        returns_to_go: Tensor,
        timesteps: Tensor,
        size: int,
        tokens: int = 0,
    ) -> Tensor:
        Tensor.training = True

        states = states[:size]
        actions = actions[:size]
        returns_to_go = returns_to_go[:size]
        timesteps = timesteps[:size]

        out = model(states, actions, returns_to_go, timesteps)

        loss = lossfn(out, actions[:size])

        optim_group.zero_grad()

        loss.backward()

        optim_group.step()
        scheduler_group.step(Tensor([tokens], dtype=dtypes.int32))

        return loss.realize()

    # ** Training Loop **
    print("Training model...")
    tokens: int = 0
    for epoch in range(epochs):
        for batch in (
            t := tqdm(
                dataset.batches(batch_size),
                "Epoch: %d, Loss: %.4f, LR: %.6f"
                % (epoch + 1, 0.0, scheduler_group.schedulers[0].get_lr().item()),
                total=math.ceil(dataset_size / batch_size),
            )
        ):
            states, actions, returns_to_go, timesteps, size = batch

            states = Tensor(states, dtype=dtypes.uint8)
            actions = Tensor(actions, dtype=dtypes.uint8)
            returns_to_go = Tensor(returns_to_go, dtype=dtypes.int8)
            timesteps = Tensor(timesteps, dtype=dtypes.uint8)

            tokens += (timesteps > 0).sum().numpy().sum()

            loss = step(states, actions, returns_to_go, timesteps, size, tokens)

            Tensor.training = False
            t.set_description(
                "Epoch: %d, Loss: %.4f, LR: %.6f"
                % (
                    epoch + 1,
                    loss.numpy(),
                    scheduler_group.schedulers[0].get_lr().item(),
                )
            )

            del batch
            del states, actions, returns_to_go, timesteps
            del loss
            gc.collect()

        # Save the model
        state_dict = get_state_dict(model)
        safe_save(
            state_dict, os.path.join(model_dir, f"model-epoch{epoch+1}.safetensors")
        )
        del state_dict
