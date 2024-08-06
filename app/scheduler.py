from tinygrad import Tensor
from tinygrad.nn.optim import OptimizerGroup
import math


class DT_Scheduler:
    def __init__(
        self,
        optimizer: OptimizerGroup,
        lr=6e-4,
        warmup_tokens: int = 512 * 20,
        final_tokens: int = 2 * 500_000 * 50,
    ):
        self.optimizer = optimizer
        self.epoch_counter = Tensor([0], requires_grad=False)

        self.lr = lr
        self.warmup_tokens = warmup_tokens
        self.final_tokens = final_tokens

        # set lr for first warmup step
        for o in self.optimizer.optimizers:
            o.lr.assign(Tensor([self.lr])).realize()

    def get_lr(self, tokens: int) -> Tensor:
        if tokens < self.warmup_tokens:
            # linear warmup
            lr_mult = float(tokens) / float(max(1, self.warmup_tokens))
        else:
            # cosine learning rate decay
            progress = float(tokens - self.warmup_tokens) / float(
                max(1, self.final_tokens - self.warmup_tokens)
            )
            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return Tensor([self.lr * lr_mult])

    def step(self, tokens: int) -> None:
        self.epoch_counter.assign(self.epoch_counter + 1).realize()
        for o in self.optimizer.optimizers:
            o.lr.assign(self.get_lr(tokens)).realize()
