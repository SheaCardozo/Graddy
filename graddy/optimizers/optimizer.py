from abc import ABC, abstractmethod

class Optimizer(ABC):
  def __init__(self, model):
    self.model = model

  def update(self, loss):
    loss.backward()
    update = self._compute_update()
    self.model.update_parameters(update)
    self.model.zero_grad()

  @abstractmethod
  def _compute_update(self, grad):
    pass

