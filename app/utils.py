from tinygrad import Tensor


def bmm(x: Tensor, y: Tensor) -> Tensor:
    return Tensor.einsum('bij,bjk->bik', x, y)
