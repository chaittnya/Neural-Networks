import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {c : i+1 for i, c in enumerate(chars)} # create a mapping from character to index, starting from 1
stoi['.'] = 0
itos = {i : c for c, i in stoi.items()}

block_size = 3
X, Y = [], []

for w in words:
    # print(w)
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        # print(''.join(itos[i] for i in context), '-->', itos[ix])
        context = context[1:] + [ix]


X = torch.tensor(X)
Y = torch.tensor(Y)
X.shape, Y.shape

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)
W1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]


sum(p.nelement() for p in parameters)

for p in parameters:
    p.requires_grad = True

for i in range(10):
    # forward pass
    emb = C[X]
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    # counts = logits.exp()
    # probs = counts / counts.sum(dim=1, keepdim=True)
    # loss = -probs[torch.arange(32), Y].log().mean()
    loss = F.cross_entropy(logits, Y) # uses optimal and fused kernels, uses mathematical optimizations and it is numerically well behaved
    print(f"Epoch {i} | loss: {loss:.4f}")
    # backward pass
    lr = 0.1
    for p in parameters:
        p.grad = None

    loss.backward()
    # update parameters
    for p in parameters:
        p.data += -lr * p.grad


# above gradient descent is too slow
# lets do mini batch gradient descent

for i in range(10000):
    # create minibatch
    ix = torch.randint(0, X.shape[0], (32,)) # (start, end, size)  <== batch size = 32

    # forward pass
    emb = C[X[ix]]  #### Only get ix rows of X
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y[ix]) # only use ix rows
    # print(f"Epoch {i} | loss: {loss:.4f}")
    # backward pass
    lr = 0.1
    for p in parameters:
        p.grad = None

    loss.backward()
    # update parameters
    for p in parameters:
        p.data += -lr * p.grad

print(f"Epoch {i+1} | loss: {loss.item():.4f}")


# testing
temb = C[X]
th = torch.tanh(temb.view(-1, 6) @ W1 + b1)
tlogits = th @ W2 + b2
tloss = F.cross_entropy(tlogits, Y)
tloss.item()

##############################################################################
# Train split, Val split, Test Split
# build the dataset
def build_dataset(words):
    block_size = 3 # context length: how many characters do we take to predict the next one?
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

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)
W1 = torch.randn((6, 300), generator=g)
b1 = torch.randn(300, generator=g)
W2 = torch.randn((300, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

for i in range(30000):
    ix = torch.randint(0, Xtr.shape[0], (256,)) # (start, end, size)  <== batch size = 32

    # forward pass
    emb = C[Xtr[ix]]  #### Only get ix rows of X
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix]) # only use ix rows

    # backward pass
    lr = 0.1
    for p in parameters:
        p.grad = None

    loss.backward()
    # update parameters
    for p in parameters:
        p.data += -lr * p.grad

print(f"Epoch {i+1} | loss: {loss.item():.4f}")

# validate
temb = C[Xdev]
th = torch.tanh(temb.view(-1, 6) @ W1 + b1)
tlogits = th @ W2 + b2
tloss = F.cross_entropy(tlogits, Ydev)
tloss.item()

# lets plot the embeddings C
plt.figure(figsize=(8,8))
plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color="white")
plt.grid("minor")

# lets try to increase embedding size of each character to increase performance
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 10), generator=g)
W1 = torch.randn((30, 300), generator=g)
b1 = torch.randn(300, generator=g)
W2 = torch.randn((300, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

for i in range(50000):
    ix = torch.randint(0, Xtr.shape[0], (256,)) # (start, end, size)  <== batch size = 32

    # forward pass
    emb = C[Xtr[ix]]  #### Only get ix rows of X
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix]) # only use ix rows

    # backward pass
    lr = 0.01
    for p in parameters:
        p.grad = None

    loss.backward()
    # update parameters
    for p in parameters:
        p.data += -lr * p.grad

print(f"Epoch {i+1} | loss: {loss.item():.4f}")

# validate
temb = C[Xdev]
th = torch.tanh(temb.view(-1, 30) @ W1 + b1)
tlogits = th @ W2 + b2
tloss = F.cross_entropy(tlogits, Ydev)
tloss.item()

########## sample names using this model
# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):

    out = []
    context = [0] * block_size # initialize with all ...
    while True:
        emb = C[torch.tensor([context])] # (1,block_size,d)
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break

    print(''.join(itos[i] for i in out))
