import math
import numpy as np

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data      = float(data)
        self.grad      = 0.0
        self._backward = lambda: None
        self._prev     = set(_children)
        self._op       = _op
        self.label     = label

    def __repr__(self):
        return f"Value(data={self.data:.4g}, grad={self.grad:.4g}, op='{self._op}', label='{self.label}')"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad  += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    __radd__ = __add__           

    def __neg__(self):
        out = Value(-self.data, (self,), 'neg')
        def _backward():
            self.grad += -1.0 * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        return Value(other) + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        if other.data == 0:
            raise ZeroDivisionError("Không thể chia cho 0")
        out = Value(self.data / other.data, (self, other), '/')
        def _backward():
            self.grad  += (1.0 / other.data) * out.grad
            other.grad += (-self.data / (other.data**2)) * out.grad
        out._backward = _backward
        return out
    def __rtruediv__(self, other):
        return Value(other) / self

    def __pow__(self, power):
        if isinstance(power, Value):
            out = Value(self.data ** power.data, (self, power), 'pow')
            def _backward():
                # d(out)/d(self)   = p * x^(p-1)
                # d(out)/d(power) = ln(x) * x^p
                self.grad   += power.data * (self.data ** (power.data - 1)) * out.grad
                power.grad  += math.log(self.data + 1e-20) * out.data * out.grad
            out._backward = _backward
        else:
            out = Value(self.data ** power, (self,), f'**{power}')
            def _backward():
                self.grad += power * (self.data ** (power - 1)) * out.grad
            out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), 'relu')
        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad     
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data + 1e-20), (self,), 'log')
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def sin(self):
        s = math.sin(self.data)
        out = Value(s, (self,), 'sin')
        def _backward():
            self.grad += math.cos(self.data) * out.grad
        out._backward = _backward
        return out

    def cos(self):
        c = math.cos(self.data)
        out = Value(c, (self,), 'cos')
        def _backward():
            self.grad += -math.sin(self.data) * out.grad
        out._backward = _backward
        return out

    def tan(self):
        t = math.tan(self.data)
        out = Value(t, (self,), 'tan')
        def _backward():
            self.grad += (1.0 + t ** 2) * out.grad   # sec^2(x)
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
