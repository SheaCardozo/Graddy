import random

from graddy.nn.module import Module
from graddy.engine.tensor import Tensor

class Conv2D(Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias = True):
    super().__init__()
    self.in_channels, self.out_channels = in_channels, out_channels

    if isinstance(kernel_size, int):
      kernel_size = (kernel_size, kernel_size)

    self.kernel_size = kernel_size
    self._bias = bias

    if isinstance(padding, int):
      padding = (padding, padding)

    self._padding = padding

    if isinstance(stride, int):
      stride = (stride, stride)
    self._stride = stride

    self.parameters["kernel"] = Tensor([[[[random.gauss(mu=0, sigma=1) for i in range(self.kernel_size[0])] for j in range(self.kernel_size[1])] for k in range(self.in_channels)] for l in range(self.out_channels)])
    
    if self._bias:
      self.parameters["bias"] = Tensor([random.gauss(mu=0, sigma=1) for i in range(self.out_channels)])

  def _pad(self, x):
    new_x = []

    for xc in x:
      new_c = []
      for p in range(self._padding[0]):
        new_c.append([0 for ph in range(2*self._padding[1] + x.shape[2])])

      for xw in xc:
        new_c.append([0 for ph in range(self._padding[1])] + [xh for xh in xw] + [0 for ph in range(self._padding[1])])

      for p in range(self._padding[0]):
        new_c.append([0 for ph in range(2*self._padding[1] + x.shape[2])])

      new_x.append(new_c)

    return Tensor(new_x)

  def _get_slice(self, sample, h, w):
    slc = []

    for channel in sample:
      new_channel = []

      for i in range(self.kernel_size[0]):
        new_row = []

        for j in range(self.kernel_size[1]):
          new_row.append(channel[h+i, w+j])

        new_channel.append(new_row)

      slc.append(new_channel)
    
    return Tensor(slc)

  
  def __call__(self, x):

    if len(x.shape) != 4:
      raise ValueError("Expects (N, C, H, W)!")
  
    if x.shape[1] != self.in_channels:
      raise ValueError(f"Input channel {x.shape[1]} mismatch with {self.in_channels}!")
    
    x = Tensor([self._pad(xt) for xt in x])

    output = []

    for n, sample in enumerate(x):
      sample_channels = []

      for channel, kernel in enumerate(self.parameters["kernel"]):
        new_channel = []
        
        for h in range(0, x.shape[2], self._stride[0]):

          if (h + self.kernel_size[0]) > x.shape[2]:
            continue 

          new_height = []
          for w in range(0, x.shape[3], self._stride[1]):

            if (w + self.kernel_size[1]) > x.shape[3]:
              continue

            slc = self._get_slice(sample, h, w)
            conv_out = sum(sum(sum(kernel * slc)))

            if self._bias:
              conv_out += self.parameters["bias"][channel]

            new_height.append(conv_out)

          new_channel.append(new_height)

        sample_channels.append(new_channel)

        print(Tensor(sample_channels).shape)

      output.append(sample_channels)
      print(Tensor(output).shape)

    return Tensor(output)
