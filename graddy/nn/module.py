from abc import ABC, abstractmethod

from graddy.engine.tensor import Tensor

class Module(ABC):
  def __init__(self):
    self.parameters = {}

  @abstractmethod
  def __call__(self, x):
    pass

  @property
  def grad(self):
    return {k: v.grad for k, v in self.parameters.items()}
  
  def zero_grad(self):
      for k, v in self.parameters.items():
        v.zero_grad()

  def update_parameters(self, update):
    for k, v in update:
      if isinstance(self.parameters[k], Tensor):
        self.parameters[k] += v
      else:
        self.parameters[k].update_parameters(v)
