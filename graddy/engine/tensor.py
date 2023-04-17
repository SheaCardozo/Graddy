from graddy.engine.item import Item

class Tensor():
  def __init__(self, data):
    if not isinstance(data, (list, Tensor)):
      raise ValueError("Tensors in Graddy are constructed from nested lists!")

    self.shape = []

    probe = data

    while isinstance(probe, (list, Tensor)) and len(probe) > 0:
      self.shape.append(len(probe))
      probe = probe[0]

    if isinstance(data[0], (list, Tensor)):
      self.data = [d if isinstance(d, Tensor) else Tensor(d) for d in data]
    else:
      self.data = [d if isinstance(d, Item) else Item(d) for d in data]

  def __repr__(self) -> str:
    return f"{self.data}"

  def __len__(self):
    return len(self.data)

  def __getitem__(self, ind):
    if isinstance(ind, int) :
      return self.data[ind]
    elif len(ind) == 1:
      return self.data[ind[0]]
    else:
      return self.data[ind[0]][ind[1:]]

  @property
  def grad(self):
    new_data = [v.grad for v in self.data]
    return Tensor(new_data) 

  def detach(self):
    for v in self.data:
      v.detach()
    return self

  # Elementwise Operations
  def __add__(self, other):
    if isinstance(other, Tensor):
      if self.shape != other.shape:
        raise ValueError(f"Shape mismatch: {self.shape} and {other.shape}")

      new_data = [row + other.data[i] for i, row in enumerate(self.data)]
    else:
      new_data = [row + other for i, row in enumerate(self.data)]

    return Tensor(new_data)

  def __mul__(self, other):
    if isinstance(other, Tensor):
      if self.shape != other.shape:
        raise ValueError(f"Shape mismatch: {self.shape} and {other.shape}")

      new_data = [row * other.data[i] for i, row in enumerate(self.data)]
    else:
      new_data = [row * other for i, row in enumerate(self.data)]

    return Tensor(new_data)

  def __pow__(self, other):
    if isinstance(other, Tensor):
      if self.shape != other.shape:
        raise ValueError(f"Shape mismatch: {self.shape} and {other.shape}")

      new_data = [row ** other.data[i] for i, row in enumerate(self.data)]
    else:
      new_data = [row ** other for i, row in enumerate(self.data)]

    return Tensor(new_data)

  def __abs__(self):
    new_data = [abs(v) for v in self.data]
    return Tensor(new_data)
  
  def relu(self):
    new_data = [v.relu() for v in self.data]
    return Tensor(new_data)

  def log(self, base=None):
    new_data = [v.log(base) for v in self.data]
    return Tensor(new_data)
  
  def zero_grad(self):
      for v in self.data:
        v.zero_grad()

  # Implemented using the above
  def __neg__(self):
    return self * -1

  def __pos__(self):
    return self

  def __radd__(self, other):
      return self + other

  def __sub__(self, other):
      return self + (-other)

  def __rsub__(self, other):
      return other + (-self)

  def __rmul__(self, other):
      return self * other

    # Matmult

  def __matmul__(self, other):
    if len(self.shape) != 2 or len(other.shape) != 2:
      raise ValueError(f"Matmul only support tensors of dim 2")

    if self.shape[1] != other.shape[0]:
      raise ValueError(f"Matmul shape mismatch: {self.shape} and {other.shape}")
    
    new_data = []
    for i in range(self.shape[0]):
      new_data.append([])
      for j in range(other.shape[1]):
        new_data[-1].append(sum([self.data[i][p] * other.data[p][j] for p in range(self.shape[1])]))
        
    return Tensor(new_data)

  def item(self):
    if len(self.data) != 1 or len(self.data[0]) != 1:
      raise RuntimeError("Item only works on Tensor with one value!")

    return self.data[0][0]