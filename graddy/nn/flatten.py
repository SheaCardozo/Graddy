import random

from graddy.nn.module import Module
from graddy.engine.tensor import Tensor

class Flatten(Module):
  def __init__(self):
    super().__init__()

  def _flatten(self, x):

    if not isinstance(x, Tensor):
      return x
    else:
      if len(x) == 1:
        return self._flatten(x[0])
      else:
        return [self._flatten(xt) for xt in x]


  def __call__(self, x):
    x = self._flatten(x)
    return x
