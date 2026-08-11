import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt').read().splitlines()
words[:8]

chars = sorted(list(set(''.join(words))))
stoi = {c : i+1 for i, c in enumerate(chars)} # create a mapping from character to index, starting from 1
stoi['.'] = 0
itos = {i : c for c, i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)

import random
random.seed(42)
random.shuffle(words)

##### Wavenet

# Train split, Val split, Test Split
# build the dataset
block_size = 8 # context length: how many characters do we take to predict the next one?
def build_dataset(words):
    X, Y = [], []
    for w in words:

        #print(w)
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            #print(''.join(itos[i] for i in context), '--->', itos[ix])
            context = context[1:] + [ix] # crop and append

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])        # 80%
Xdev, Ydev = build_dataset(words[n1:n2])    # 10%
Xte, Yte = build_dataset(words[n2:])        # 10%

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x: torch.Tensor):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1D:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.dim = dim
        self.training = True

        # parameters (trained with backpropagation)
        self.gamma = torch.ones((1, dim)) # gain
        self.beta = torch.zeros((1, dim)) # bias

        # buffers (updated using momentum)
        self.running_mean = torch.zeros((1, dim))
        self.running_var = torch.ones((1, dim))

    def __call__(self, x: torch.Tensor):
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            xmean = x.mean(dim=dim, keepdim=True) # batch mean
            xvar = x.var(dim=dim, keepdim=True, unbiased=True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var

        x_hat = (x - xmean) / torch.sqrt(xvar + self.eps)

        self.out = self.gamma * x_hat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim)) # this is the 'C'

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]


class FlattenConsecutive:
    def __init__(self, n):
        # n = number of elements that are consecutive and we want to concatenate in last dim
        self.n = n
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n) # expected #(B, T//n, C*n)
        if x.shape[1] == 1: # when all the pairing of all levels done, this dim will become 1 |  exmplained later
            x = x.squeeze(dim=1)
        self.out = x
        return self.out

    def parameters(self):
        return []

class Sequential:
    def __init__(self, layers):
        self.layers = layers
        with torch.no_grad():
            self.layers[-1].weight *= 0.1

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

n_embed = 24 # embedding vector dim
n_hidden = 128 # number of neurons in hidden layer

model = Sequential([
    Embedding(vocab_size, n_embed),
    FlattenConsecutive(2), # group of 2-2
    Linear(n_embed  * 2, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),

    FlattenConsecutive(2), # group of 4-4
    Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),

    FlattenConsecutive(2), # group of 8-8
    Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),

    Linear(n_hidden, vocab_size)
])



parameters = model.parameters()
print("Number of parameters :", sum(p.nelement() for p in parameters))

for p in parameters:
    p.requires_grad = True

# ---- sanity-check the shapes flowing through the hierarchical model ----
ix = torch.randint(0, Xtr.shape[0], (6, )) # lets take batch of 4 examples
Xb, yb = Xtr[ix], Ytr[ix]
logits = model(Xb)
print(Xb.shape)
Xb

logits.shape

model.layers[0].out.shape # output of embedding layer

model.layers[1].out.shape # output of flatten layer

model.layers[2].out.shape # output of linear layer

print((torch.randn((6, 80)) @ torch.randn((80, 200)) + torch.randn(200)).shape) # similar to linear layer

print((torch.randn((6, 4, 80)) @ torch.randn((80, 200)) + torch.randn(200)).shape)

print((torch.randn((6, 8, 4, 80)) @ torch.randn((80, 200)) + torch.randn(200)).shape)

#  1  2  3  4  5  6  7  8   Shape: (B, 8, 10)

# (1 2) (3 4) (5 6) (7 8)    <-- we want to process all the 4 dimentions in parallel like a batch | Shape: (B, 4, 20)

# # (1 2 3 4) (5 6 7 8)    Shape: (B, 2, 40)

# #  (1 2 3 4 5 6 7 8)    Shape: (B, 1, 80)  <--- euivalent to current emb: (torch.randn((6, 80))


print((torch.randn((6, 4, 20)) @ torch.randn((20, 200)) + torch.randn(200)).shape)

# we need to change our flatten layer, we want (6, 4, 20) for following
e = torch.randn(6, 8, 10)

list(range(10))[::2] # evens

list(range(10))[1::2] # odds

explicit = torch.cat([e[:, ::2, :], e[:, 1::2, :]], dim=2)
explicit.shape

(e.view(6, 4, 20) == explicit).all()

# training

max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    # minibactch
    ix = torch.randint(0, Xtr.shape[0], (batch_size, ))
    Xb, yb = Xtr[ix], Ytr[ix]

    # Forward Pass
    logits = model(Xb)

    loss = F.cross_entropy(logits, yb) # loss function

    # backard pass
    for p in parameters:
        p.grad = None

    loss.backward()

    # update: simple SGD
    lr = 0.1 if i < 150000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())


# put layers into eval mode
for layer in model.layers:
    layer.training = False

plt.plot(torch.tensor(lossi).view(-1, 1000).mean(dim=1))

splits = {
    'train': (Xtr, Ytr),
    'val': (Xdev, Ydev),
    'test': (Xte, Yte),
}

@torch.no_grad() # disable gradient tracking
def split_loss(split):
    x, y = splits[split]
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss('train')
split_loss('val')

model.layers[3].running_mean.shape

# Problem
# batch norm input is (32, 4, 68)

for layer in model.layers:
    print(layer.__class__.__name__, ":", tuple(layer.out.shape))

e = torch.randn(32, 4, 68)
emean = e.mean(dim=0, keepdim=True) #(1, 4, 68) --> intended (1, 1, 68)
evar = e.var(dim=0, keepdim=True) #(1, 4, 68) --> intended (1, 1, 68)
ehat = (e - emean) / torch.sqrt(evar+1e-5) # (32, 4, 68)
ehat.shape

e = torch.randn(32, 4, 68)
emean = e.mean(dim=(0, 1), keepdim=True) #(1, 1, 68)
evar = e.var(dim=(0, 1), keepdim=True) #(1, 1, 68)
ehat = (e - emean) / torch.sqrt(evar+1e-5) # (32, 4, 68)
print(emean.shape)
print(emean.shape)
ehat.shape

with torch.no_grad():
    for _ in range(20):
        out = []
        context = [0] * block_size # initialize with all ...
        while True:
            # forward pass
            logits = model(torch.tensor([context]))
            probs = F.softmax(logits, dim=1)
            # sample from the distribution
            ix = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break

        print(''.join(itos[i] for i in out))

for layer in model.layers:
    print(layer.__class__.__name__, ":", tuple(layer.out.shape))

# Problem
# Batch norm is getting
