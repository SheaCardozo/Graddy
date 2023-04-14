import math

from graddy.engine.tensor import Tensor
from graddy.nn.module import Module

class ReLU(Module):
  def __init__(self):
    super().__init__()

  def __call__(self, x):
    return x.relu()

class SoftMax(Module):
  def __init__(self):
    super().__init__()

  def __call__(self, x):

    x = self._probe(x)
    return x

  def _probe(self, xt):
    if isinstance(xt[0], Tensor):
      return Tensor([self._probe(xts) for xts in xt])
    else:
      max_v = max(xt)
      denom = sum([math.e ** (v - max_v) for v in xt])
      return Tensor([math.e ** (v - max_v) / denom for v in xt])