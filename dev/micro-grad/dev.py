import math
import random
import numpy as np
import matplotlib.pyplot as plt

from graphviz import Digraph

def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
    
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name = uid + n._op, label = n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)
            
    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        
    return dot


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


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1), label=f'w{i}') for i in range(nin)]
        self.b = Value(random.uniform(-1, 1), label='b')

    def __call__(self, x):
        activation = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = activation.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        # nout = number of neurons in the layer
        # nin = number of inputs to the layer(how many inputs each neuron will have)
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs # return a list of outputs if there are multiple neurons in the layer, else return the single output of the neuron

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        # nin = number of inputs to the network
        # nouts = list of number of neurons in each layer
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def neuron_test():
    # l = Layer(3, 2)  # 3 inputs, 2 neurons in the layer
    x = [1.0, 2.0, 3.0]
    nw = MLP(3, [4, 4, 1]) # neural network with 3 inputs, 3 layers with 4, 4, and 1 neurons respectively
    y = nw(x)

    ###### Simple classification dataset example ######
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]

    ys = [1.0, -1.0, -1.0, 1.0] # labels for the dataset
    ypred = [nw(x) for x in xs] # predictions for the dataset
    print(ypred)

    loss = sum((yout - ygt)**2 for yout, ygt in zip(ypred, ys)) # mean squared error loss
    print("Loss:", loss)

    loss.backward() # backpropagation to compute gradients

    print(nw.layers[0].neurons[0].w[0].grad)
    print(nw.layers[0].neurons[0].w[0].data)

    # print(nw.parameters())

    # draw_dot(loss).render('graph', view=True)

    for p in nw.parameters():
        p.data += -0.01 * p.grad # update parameters using gradient descent

    print(nw.layers[0].neurons[0].w[0].grad)
    print(nw.layers[0].neurons[0].w[0].data)
    ypred = [nw(x) for x in xs]
    loss = sum((yout - ygt)**2 for yout, ygt in zip(ypred, ys)) # mean squared error loss
    print("Loss after param update:", loss)


    # Training Loop
    for i in range(30):
        for p in nw.parameters():
            p.grad = 0.0 # reset gradients to zero before backpropagation

        loss.backward()

        for p in nw.parameters():
            p.data += -0.01 * p.grad # update parameters using gradient descent

        ypred = [nw(x) for x in xs]
        loss = sum((yout - ygt)**2 for yout, ygt in zip(ypred, ys)) # mean squared error loss
        print("Loss after param update:", loss)

    print("Predictions after training:", ypred)

neuron_test()

# a = Value(2.0, label='a')
# b = Value(4.0, label='b')
# print(a-b)

def test():
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b
    e.label = 'e' 
    d = e + c
    d.label = 'd'

    f = Value(-2.0, label='f')

    L = d * f
    L.label = 'L'
    L.grad = 1.0 # this will always be 1 as dL/dL = 1, so we can set it to 1.0 to start the backpropagation process

    '''
    L = d * f

    derivative: (f(x+h)- f(x)) / h

    diff wrt d
    ((d+h)*f - d*f) / h => d*f + h*f - d*f / h => h*f / h => f 
    '''

    f.grad = d.data
    d.grad = f.data

    '''
    dL/dc = dL/dd * dd/dc

    dd/dc = ?
    d = c + e

    derivative: (f(x+h)- f(x)) / h
    diff wrt c

    ((c+h) + e - (c+e)) / h => (c + h + e - c - e) / h => h / h => 1
    dd/dc = 1
    dL/dc = dL/dd * dd/dc => dL/dd * 1 => dL/dc => f 
    '''
    c.grad = d.grad * 1.0 # d.grad = f.data
    e.grad = d.grad * 1.0 # d.grad = f.data

    '''
    a.grad = dL/de * de/da
    de/da = ?
    e = a * b
    derivative: (f(x+h)- f(x)) / h
    diff wrt a
    ((a+h)*b - a*b) / h => (a*b + h*b - a*b) / h => h*b / h => b
    de/da = b
    a.grad = dL/de * de/da => e.grad * b.data
    '''
    a.grad = e.grad * b.data
    b.grad = e.grad * a.data

    # draw_dot(L).render('graph', view=True)

    def check():
        h = 0.0001
        a = Value(2.0, label='a')
        b = Value(-3.0, label='b')
        c = Value(10.0, label='c')
        e = a * b
        e.label = 'e' 
        d = e + c
        d.label = 'd'
        f = Value(-2.0, label='f')
        L = d * f
        L.label = 'L'
        L1 = L.data

        a = Value(2.0 + h, label='a')
        b = Value(-3.0, label='b')
        c = Value(10.0, label='c')
        e = a * b
        e.label = 'e' 
        d = e + c
        d.label = 'd'
        f = Value(-2.0, label='f')
        L = d * f
        L.label = 'L'
        L2 = L.data

        print((L2 - L1) / h)



    ######################### Simple neural network example #########################

    # input
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    # weights
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    # bias
    b = Value(6.8813735870195432, label='b')

    x1w1 = x1 * w1; x1w1.label = 'x1*w1'
    x2w2 = x2 * w2; x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
    n = x1w1x2w2 + b; n.label = 'n'
    o = n.tanh(); o.label = 'o'


    # backpropagation
    o.grad = 1.0

    '''
    o = tanh(n)
    do/dn = ?

    d tanh(x) / dx = 1 - tanh^2(x)

    do/dn = 1 - (tanh(n))**2 = 1 - o**2
    '''

    # 1 - o.data**2 = 0.5
    n.grad = 0.5

    x1w1x2w2.grad = 0.5 # n.grad * 1.0 == 0.5 * dn/d(x1w1x2w2)  <== dn/d(x1w1x2w2) = 1 
    b.grad = 0.5 # n.grad * 1.0 # 0.5 * dn/db <== dn/db = 1

    x1w1.grad = 0.5 # x1w1x2w2.grad * 1.0 == 0.5 * d(x1w1x2w2)/d(x1w1) <== d(x1w1x2w2)/d(x1w1) = 1
    x2w2.grad = 0.5 # x1w1x2w2.grad * 1.0 == 0.5 * d(x1w1x2w2)/d(x2w2) <== d(x1w1x2w2)/d(x2w2) = 1

    x2.grad = x2w2.grad * w2.data # 0.5 * d(x2w2)/dx2 <== d(x2w2)/dx2 = w2.data
    w2.grad = x2w2.grad * x2.data # 0.5 * d(x2w2)/dw2 <== d(x2w2)/dw2 = x2.data
    x1.grad = x1w1.grad * w1.data # 0.5 * d(x1w1)/dx1 <== d(x1w1)/dx1 = w1.data
    w1.grad = x1w1.grad * x1.data # 0.5 * d(x1w1)/dw1 <== d(x1w1)/dw1 = x1.data

    # draw_dot(o).render('graph2', view=True)


    ################### Automatic backpropagation example ###################

    # input
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    # weights
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    # bias
    b = Value(6.8813735870195432, label='b')

    x1w1 = x1 * w1; x1w1.label = 'x1*w1'
    x2w2 = x2 * w2; x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
    n = x1w1x2w2 + b; n.label = 'n'

    # o = n.tanh(); o.label = 'o'
    e = (2*n).exp(); e.label = 'e' # e^2n
    o = (e-1) / (e+1); o.label = 'o' # tanh(n) = (e^2n - 1) / (e^2n + 1)

    o.backward()
    # o._backward()
    # n._backward()
    # x1w1x2w2._backward()
    # x1w1._backward()
    # x2w2._backward()

    draw_dot(o).render('graph2', view=True)


# test()