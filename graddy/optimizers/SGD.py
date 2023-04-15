from graddy.optimizers.optimizer import Optimizer
from graddy.optimizers.utils import update_tree_elementwise, update_tree_paired

class SGD(Optimizer):
  def __init__(self, model, lr):
    super().__init__(model)
    self.lr = lr

  def _compute_update(self):
    update = update_tree_elementwise(self.model.grad, lambda u: -self.lr*u)
    return update

class SGDwithMomentum(Optimizer):
  def __init__(self, model, lr, momentum):
    super().__init__(model)
    self.lr = lr
    self.momentum = momentum
    self.last_update = None
    
  def _compute_update(self):
    if self.last_update is None:
      self.last_update = update_tree_elementwise(self.model.grad, lambda u: -self.lr*u)
    else:
      self.last_update = update_tree_paired(self.model.grad, self.last_update, lambda u, v: -self.lr*u + self.momentum*v)
    return self.last_update
