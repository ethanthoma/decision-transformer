from tinygrad.nn.optim import AdamW, OptimizerGroup
from tinygrad.nn.state import get_state_dict


class DT_Optimizer(OptimizerGroup):
    def __init__(
        self,
        model,
        b1: float = 0.9,
        b2: float = 0.95,
        lr: float = 6e-4,
        weight_decay: float = 0.1,
    ):
        weight_decay_list = []
        no_weight_decay_list = []

        for k, v in get_state_dict(model).items():
            if "norm" in k or "bias" in k or "embed_a" in k:
                no_weight_decay_list.append(v)
            else:
                weight_decay_list.append(v)

        weight_decay_optim = AdamW(
            weight_decay_list, lr=lr, b1=b1, b2=b2, weight_decay=weight_decay
        )
        no_weight_decay_optim = AdamW(
            no_weight_decay_list, lr=lr, b1=b1, b2=b2, weight_decay=0.0
        )

        super().__init__(weight_decay_optim, no_weight_decay_optim)
