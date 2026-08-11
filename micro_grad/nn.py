from micrograd.engine import Value
import random


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []



# Defines a single neuron in the network
class Neuron(Module):
    def __init__(self, nin, nonlinearity=True):
        self.w = [Value(random.uniform(-1, 1), label=f'w{i}') for i in range(nin)]
        self.b = Value(random.uniform(-1, 1), label='b')
        self.nonlinearity = nonlinearity

    def __call__(self, x):
        activation = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        if self.nonlinearity:
            out = activation.relu()
        else:
            out = activation
        return out

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlinearity else 'Linear'}Neuron({len(self.w)})"



# Defines a layer of neurons in the network
class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        # nout = number of neurons in the layer
        # nin = number of inputs to the layer(how many inputs each neuron will have)
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs # return a list of outputs if there are multiple neurons in the layer, else return the single output of the neuron

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer [{', '.join(str(n) for n in self.neurons)}]"



# Defines a multi-layer perceptron (MLP) network
class MLP(Module):
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

    def __repr__(self):
        return f"MLP [{', '.join(str(layer) for layer in self.layers)}]"
