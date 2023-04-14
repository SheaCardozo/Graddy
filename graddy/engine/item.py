class Item():
  def __init__(self, x, _children = ()):
    self.value = float(x)
    self.grad = 0.
    self._children = set(_children)
    self._backward = lambda: None

  def __repr__(self) -> str:
    return f"Item(Value={self.value}, Grad={self.grad})"

  def clear_grad(self):
    self.grad = 0.
    for child in self._children:
      child.clear_grad()

  def backward(self):
    self.clear_grad()

    graph = []
    visited = set()

    def build_graph(node):
      if node not in visited:
        visited.add(node)
        for child in node._children:
            build_graph(child)
        graph.append(node)

    build_graph(self)

    self.grad = 1.
    for node in reversed(graph):
      node._backward()

  # Backprop Operations
  def __add__(self, other):
      if not isinstance(other, Item):
        other = Item(other)

      out = Item(self.value + other.value, (self, other))
      
      def _backward():
        self.grad += out.grad
        other.grad += out.grad
          
      out._backward = _backward
      return out 

  def __mul__(self, other):
      if not isinstance(other, Item):
        other = Item(other)

      out = Item(self.value * other.value, (self, other))
      
      def _backward():
        self.grad += other.value * out.grad
        other.grad += self.value * out.grad

      out._backward = _backward
      return out 


  def __pow__(self, other):
      if not isinstance(other, Item):
        other = Item(other)

      out = Item(self.value ** other.value, (self, other))
      
      def _backward():
        self.grad += (other.value * self.value**(other.value-1)) * out.grad

      out._backward = _backward
      return out 

  # Boolean Comparison Operators (not used for backprop)
  def __lt__(self, other):
    if not isinstance(other, Item):
      other = Item(other)
    return self.value < other.value

  def __le__(self, other):	
    if not isinstance(other, Item):
      other = Item(other)
    return self.value <= other.value

  def __gt__(self, other):
    if not isinstance(other, Item):
      other = Item(other)
    return self.value > other.value

  def __ge__(self, other):
    if not isinstance(other, Item):
      other = Item(other)
    return self.value >= other.value

  # Implemented using the above
  def __neg__(self):
    return self * -1

  def __pos__(self):
    return self

  def __abs__(self):
    return (-self) if self < 0 else self

  def __rpow__(self, other):
    if not isinstance(other, Item):
      other = Item(other)

    return other ** self

  def __radd__(self, other):
      return self + other

  def __sub__(self, other):
      return self + (-other)

  def __rsub__(self, other):
      return other + (-self)

  def __rmul__(self, other):
      return self * other

  def __truediv__(self, other):
      return self * other**-1

  def __rtruediv__(self, other):
      return other * self**-1