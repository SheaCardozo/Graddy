import random

from graddy.nn.module import Module
from graddy.engine.tensor import Tensor

class Linear(Module):
  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.in_dim, self.out_dim = in_dim, out_dim
    self.parameters["ff"] = Tensor([[random.gauss(mu=0, sigma=1) for i in range(self.out_dim)] for j in range(self.in_dim)])

  def __call__(self, x):
    x = x @ self.parameters["ff"]
    return x
