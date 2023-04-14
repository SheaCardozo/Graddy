from graddy.nn.module import Module

class SequentialModel(Module):
  def __init__(self, layers):
    super().__init__()

    self.n_layers = len(layers)

    for i, layer in enumerate(layers):
      self.parameters[i] = layer

  def __call__(self, x):
    for i in range(self.n_layers):
      x = self.parameters[i](x)
    return x
