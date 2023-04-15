from graddy.engine.tensor import Tensor

def update_tree_elementwise (d, op):
  return {k: (op(v) if isinstance(v, Tensor) else update_tree_elementwise(v, op)) for k, v in d.items()}

def update_tree_paired (d, u, op):
  return {k: (op(v, u[k]) if isinstance(v, Tensor) else update_tree_paired(v, u[k], op)) for k, v in d.items()}