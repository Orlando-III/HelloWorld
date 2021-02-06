import numpy as np
import nn
import nnfs
from nnfs.datasets import spiral_data

X = np.random.randn(1, 5)

net = nn.Network([5, 5, 5, 5, 5, 3])
print(net.forward_prop(X))
