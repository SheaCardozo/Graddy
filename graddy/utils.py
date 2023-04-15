from graddy.engine.tensor import Tensor

def update_tree_elementwise (d, op):
  return {k: (op(v) if isinstance(v, Tensor) else update_tree_elementwise(v, op)) for k, v in d.items()}