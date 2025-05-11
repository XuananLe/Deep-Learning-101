from nanograd import Value
from typing import List
import numpy as np
import random

class Neuron(object):
    def __init__(self, d_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(d_in)]
        self.b = Value(random.uniform(-1, 1))

    def parameters(self):
        return self.w + [self.b]

    def __call__(self, x):
        raw_act = np.dot(self.w, x)    
        out = raw_act.tanh()
        return out
    
class Layer(object):
    # nout la so neuron lop do !
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        
class MLP(object):
    def __init__(self, n_in, n_layers: List[int]):
        self.sz = [n_in] + n_layers
        # 3, 4, 4, 1
        self.layers = [Layer(self.sz[i], self.sz[i+1]) for i in range(len(n_layers))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def mse_loss(ys, ypred):
    return np.mean((ys - ypred) ** 2)

lr = 0.01
xs = np.random.random((4, 3))
mlp = MLP(3, [4,4,1])
ys = np.array([-1,-1,1,1])
ypred = [mlp(x) for x in xs]
loss  = mse_loss(ys, ypred)
print(loss)

# Backprop
loss.backward()

# One step of gradient descent
for _ in range(10):
    for p in mlp.parameters():
        p.data -= lr * p.grad
        
    ypred = [mlp(x) for x in xs]
    loss  = mse_loss(ys, ypred)

print("Final loss")
print(loss)

print("Final prediction")
ypred = [mlp(x) for x in xs]
print(ypred)