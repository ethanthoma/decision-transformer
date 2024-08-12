from tinygrad import Tensor
from tinygrad.nn.optim import Optimizer
from tinygrad.dtype import dtypes
import math


class LR_Scheduler:
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.epoch_counter = Tensor(
            [0], requires_grad=False, device=self.optimizer.device
        )

    def get_lr(self) -> Tensor:
        return Tensor([0], dtype=dtypes.float)

    def step(self, *args) -> None:
        self.epoch_counter.assign(self.epoch_counter + 1).realize()
        self.optimizer.lr.assign(self.get_lr()).realize()


class LRSchedulerGroup:
    def __init__(self, *schedulers: LR_Scheduler):
        self.schedulers = schedulers

    def step(self, *args) -> None:
        for s in self.schedulers:
            s.step(*args)


class DT_Scheduler(LR_Scheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        lr: float = 6e-4,
        warmup_tokens: int = 512 * 20,
        final_tokens: int = 2 * 500_000 * 50,
    ):
        super().__init__(optimizer)

        self.lr = lr
        self.tokens = Tensor([0], requires_grad=False, device=self.optimizer.device)
        self.warmup_tokens = min(warmup_tokens, final_tokens)
        self.final_tokens = final_tokens

        self.optimizer.lr.assign(self.get_lr()).realize()

    def get_lr(self) -> Tensor:
        tokens = self.tokens.item()
        if tokens < self.warmup_tokens:
            # linear warmup
            lr_mult = float(tokens) / float(max(1, self.warmup_tokens))
        else:
            # cosine learning rate decay
            progress = float(tokens - self.warmup_tokens) / float(
                max(1, self.final_tokens - self.warmup_tokens)
            )
            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return Tensor([self.lr * lr_mult], dtype=dtypes.float)

    def step(self, tokens: Tensor) -> None:
        self.tokens.assign(tokens).realize()
        super().step()
