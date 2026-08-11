import math


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        # assume by default changing a value does not change the output of the function, so the default gradient is 0
        self.grad = 0.0
        self._backward = lambda: None # function to compute the gradient of this value with respect to its inputs
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = None
        if isinstance(other, Value):
            out = Value(self.data + other.data, _children=(self, other), _op='+')
        else:
            out = Value(self.data + other, _children=(self,), _op='+')

        def backward():
            self.grad += out.grad * 1.0 # one neuron can be connected to multiple neurons, so we need to accumulate the gradients from all the neurons that are connected to this neuron
            if isinstance(other, Value):
                other.grad += out.grad * 1.0

        out._backward = backward
        return out

    def __radd__(self, other): # other + self
        return self + other

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)


    def __mul__(self, other):
        out = None
        if isinstance(other, Value):
            out = Value(self.data * other.data, _children=(self, other), _op='*')
        else:
            out = Value(self.data * other, _children=(self,), _op='*')

        def backward():
            if isinstance(other, Value):
                self.grad += out.grad * other.data
                other.grad += out.grad * self.data
            else:
                self.grad += out.grad * other

        out._backward = backward
        return out

    def __rmul__(self, other): # other * self
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data ** other, _children=(self,), _op=f'**{other}')
        def backward():
            self.grad += out.grad * (other * (self.data ** (other - 1)))
        out._backward = backward
        return out

    '''
        a / b
         = a * (1/b)
         = a * (b ** -1)
    '''
    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other): # other / self
        return other * (self ** -1)

    def exp(self):
        x = self.data
        out = Value(math.exp(x), _children=(self,), _op='exp')

        def backward():
            self.grad += out.grad * out.data # d/dx e^x = e^x ==> out.data = e^x
        out._backward = backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, _children=(self,), _op='tanh')

        def backward():
            self.grad += out.grad * (1 - t**2) # d/dx tanh(x) = 1 - tanh^2(x)

        out._backward = backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # Get the topological order of the nodes in the computation graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
