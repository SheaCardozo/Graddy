from graddy.nn.module import Module
from graddy.engine.tensor import Tensor

class Flatten(Module):
  def __init__(self, start_dim=0):
    super().__init__()
    self.start_dim = start_dim

  def _flatten(self, x, count):

    if not isinstance(x, Tensor):
      return x
    else:
      if len(x) == 1 and count >= self.start_dim:
        return self._flatten(x[0], count+1)
      else:
        return Tensor([self._flatten(xt, count+1) for xt in x])


  def __call__(self, x):
    x = self._flatten(x, count=0)
    return x
