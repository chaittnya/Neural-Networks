import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt').read().splitlines()
words[:8]

len(words)

chars = sorted(list(set(''.join(words))))
stoi = {c : i+1 for i, c in enumerate(chars)} # create a mapping from character to index, starting from 1
stoi['.'] = 0
itos = {i : c for c, i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)

# Train split, Val split, Test Split
# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?
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

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])        # 80%
Xdev, Ydev = build_dataset(words[n1:n2])    # 10%
Xte, Yte = build_dataset(words[n2:])        # 10%

splits = {
    'train': (Xtr, Ytr),
    'val': (Xdev, Ydev),
    'test': (Xte, Yte),
}

@torch.no_grad() # disable gradient tracking
def split_loss(split):
    x, y = splits[split]
    emb = C[x]  # (N, block_size, n_embed). block_size=context length
    emb_cat = emb.view(emb.shape[0], -1) # (N, block_size * n_embed)
    z1 = emb_cat @ W1 + b1 # (N, n_hidden)
    h = torch.tanh(z1) # (N, n_hidden)
    logits = h @ W2 + b2 # (N, vocab_size)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

# =====================================================================
# Stage 1: baseline (bad) initialization
# =====================================================================
n_embed = 10 # the dimentionality of the character embedding vectors
n_hidden = 200 # the number of neuron in hidden layer

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embed),              generator=g)

W1 = torch.randn((n_embed * block_size, n_hidden),  generator=g)
b1 = torch.randn(n_hidden,                          generator=g)

W2 = torch.randn((n_hidden, vocab_size),            generator=g)
b2 = torch.randn(vocab_size,                        generator=g)

parameters = [C, W1, b1, W2, b2]
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

max_steps = 200
batch_size = 32
lossi = []

for i in range(max_steps):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g) # (start, end, size)  <== batch size = 32
    Xb, Yb = Xtr[ix], Ytr[ix] # batch

    # forward pass
    emb = C[Xb]  # embed characters into vectors
    emb_cat = emb.view(emb.shape[0], -1) # concatenate context vectors in single instance together
    z1 = emb_cat @ W1 + b1 # hidden pre activation
    h = torch.tanh(z1) # hidden layer
    logits = h @ W2 + b2 # output layer
    loss = F.cross_entropy(logits, Yb) # loss function

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update parameters
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')

    lossi.append(loss.log10().item())

print(f"Epoch {i+1} | loss: {loss.item():.4f}")

plt.plot(lossi)

split_loss('train')
split_loss('val')

# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)
with torch.no_grad():
    for _ in range(20):
        out = []
        context = [0] * block_size # initialize with all ...
        while True:
            # forward pass
            emb = C[torch.tensor([context])] # (1, block_size, n_embed)
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            # sample from the distribution
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break

        print(''.join(itos[i] for i in out))

-torch.tensor(1/27.0).log()

'''
There are many problems in a way we initialized out weights:

At the first step of training, the loss was very high(0/ 200000: 26.5404). We wasn't expecting that high loss.
At the start, even if we assign uniform probability to each character, loss would have been 3.29
'''

# 4-dimentional view of the issue
logits_list = [torch.tensor([0.1, 0.1, 0.1, 0.1]), torch.tensor([0.01, 0.01, 5.0, 0.01]), torch.tensor([0.01, 5.0, 0.01, 0.01]), torch.randn(4) * 10]

for logits in logits_list:
    probs = torch.softmax(logits, dim=0)
    loss = -probs[2].log() # 2 is the correct class -> loss is less if it has high val
    print(logits)
    print(probs)
    print(loss)
    print('--------------------------------------------------')

'''
tensor([-14.3433,  15.0456,   2.0128,   5.4239])
tensor([1.7241e-13, 9.9993e-01, 2.1873e-06, 6.6274e-05])
tensor(13.0328)

if logits get extreme values, we are getting very high loss, not good.

we want logits to be roughly zero and approximatly equal at the time of initialization
'''

# =====================================================================
# Stage 2: fix 1 - softmax "confidently wrong" at init
# scale down W2 and zero b2 so logits start near zero
# =====================================================================
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embed),              generator=g)

W1 = torch.randn((n_embed * block_size, n_hidden),  generator=g)
b1 = torch.randn(n_hidden,                          generator=g)

W2 = torch.randn((n_hidden, vocab_size),            generator=g) * 0.01 # scaling down w2 because logit calculation involves w2. (logits = h @ W2 + b2)
b2 = torch.randn(vocab_size,                        generator=g) * 0

parameters = [C, W1, b1, W2, b2]
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g) # (start, end, size)  <== batch size = 32
    Xb, Yb = Xtr[ix], Ytr[ix] # batch

    # forward pass
    emb = C[Xb]  # embed characters into vectors
    emb_cat = emb.view(emb.shape[0], -1) # concatenate context vectors in single instance together
    z1 = emb_cat @ W1 + b1 # hidden pre activation
    h = torch.tanh(z1) # hidden layer
    logits = h @ W2 + b2 # output layer
    loss = F.cross_entropy(logits, Yb) # loss function

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update parameters
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')

    lossi.append(loss.log10().item())

print(f"Epoch {i+1} | loss: {loss.item():.4f}")

plt.plot(lossi)

# So we can say that we are not wasting initial steps to bring down the weights.

split_loss('train')
split_loss('val')

# =====================================================================
# Stage 3: investigate tanh saturation on the softmax-fixed model
# =====================================================================
'''
The second problem is about hidden state(h).
Many values of h are either -1 or 1
'''

h

plt.figure(figsize=(20,10))
plt.imshow(h.abs() > 0.99, cmap='gray', interpolation='nearest');

'''
Here white dot means tanh of that hidden item is ~1(this value being in the flat region of tanh curve) and black means it is less than 0.99
'''

print(h.shape)
print(h.view(-1).shape)

plt.hist(h.view(-1).tolist(), 50);

plt.hist(z1.view(-1).tolist(), 50);

'''
There are lots of values far from 0, so they are curbed down to -1 or 1 by tanh

Problem: We apply tanh element wise (torch.tanh(z1))

The gradient of tanh is '(1 - t**2) * out.grad' <-- by micrograd

This will be 0 for if tanh output is near 1 or -1

So, gradient(or out.grad) will not flow backward(it vanishes)
'''

# =====================================================================
# Stage 4: fix 2 - tanh saturation at init
# scale down W1 and b1 so pre-activations don't saturate tanh
# =====================================================================
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embed),              generator=g)

W1 = torch.randn((n_embed * block_size, n_hidden),  generator=g) * 0.2
b1 = torch.randn(n_hidden,                          generator=g) * 0.01

W2 = torch.randn((n_hidden, vocab_size),            generator=g) * 0.01 # scaling down w2 because logit calculation involves w2. (logits = h @ W2 + b2)
b2 = torch.randn(vocab_size,                        generator=g) * 0

parameters = [C, W1, b1, W2, b2]
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g) # (start, end, size)  <== batch size = 32
    Xb, Yb = Xtr[ix], Ytr[ix] # batch

    # forward pass
    emb = C[Xb]  # embed characters into vectors
    emb_cat = emb.view(emb.shape[0], -1) # concatenate context vectors in single instance together
    z1 = emb_cat @ W1 + b1 # hidden pre activation
    h = torch.tanh(z1) # hidden layer
    logits = h @ W2 + b2 # output layer
    loss = F.cross_entropy(logits, Yb) # loss function

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update parameters
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')

    lossi.append(loss.log10().item())
    break

print(f"Epoch {i+1} | loss: {loss.item():.4f}")

plt.hist(h.view(-1).tolist(), 50);

plt.hist(z1.view(-1).tolist(), 50);

plt.figure(figsize=(20,10))
plt.imshow(h.abs() > 0.99, cmap='gray', interpolation='nearest');

'''
This means very less neurons saturated on -1 or 1 of tanh
'''

max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g) # (start, end, size)  <== batch size = 32
    Xb, Yb = Xtr[ix], Ytr[ix] # batch

    # forward pass
    emb = C[Xb]  # embed characters into vectors
    emb_cat = emb.view(emb.shape[0], -1) # concatenate context vectors in single instance together
    z1 = emb_cat @ W1 + b1 # hidden pre activation
    h = torch.tanh(z1) # hidden layer
    logits = h @ W2 + b2 # output layer
    loss = F.cross_entropy(logits, Yb) # loss function

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update parameters
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')

    lossi.append(loss.log10().item())

print(f"Epoch {i+1} | loss: {loss.item():.4f}")
plt.plot(lossi);

split_loss('train')
split_loss('val')

'''
Loss log:

original
train: 2.12
val  : 2.16

fix softmax confidently wrong
train: 2.07
val  : 2.13

fix tanh layer too saturated at init
train: 2.03
val  : 2.10
'''

# =====================================================================
# Stage 5: principled fix - Kaiming initialization
# =====================================================================
x = torch.randn(1000, 10) # 1000 example, each of dim 10
w = torch.randn(10, 200) # layer with 200 neurons with input of size 10
y = x @ w
print(x.mean(), x.std())
print(y.mean(), y.std())
plt.figure(figsize=(20, 5))
plt.subplot(121)
plt.hist(x.view(-1).tolist(), 50, density=True);
plt.subplot(122)
plt.hist(y.view(-1).tolist(), 50, density=True);

'''
Here standard deviation of y is very different(higher) than the input x

After multipying x with w, it is distorting the std/distribution of x

we want the std of y should remain approx equal to the input

what should we do, => divide w with sqrt(size of input[also called fan-in])

kaimin initialization formula = gain / sqrt(fan-in)
.. gain for tanh = 5/3
'''

x = torch.randn(1000, 10) # 1000 example, each of dim 10
w = torch.randn(10, 200) / 10**0.5 # layer with 200 neurons with input of size 10
y = x @ w
print(x.mean(), x.std())
print(y.mean(), y.std())
plt.figure(figsize=(20, 5))
plt.subplot(121)
plt.hist(x.view(-1).tolist(), 50, density=True);
plt.subplot(122)
plt.hist(y.view(-1).tolist(), 50, density=True);

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embed),              generator=g)

W1 = torch.randn((n_embed * block_size, n_hidden),  generator=g) * (5/3) / ((n_embed * block_size) ** 0.5)
b1 = torch.randn(n_hidden,                          generator=g) * 0.01

W2 = torch.randn((n_hidden, vocab_size),            generator=g) * 0.01 # scaling down w2 because logit calculation involves w2. (logits = h @ W2 + b2)
b2 = torch.randn(vocab_size,                        generator=g) * 0

parameters = [C, W1, b1, W2, b2]
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g) # (start, end, size)  <== batch size = 32
    Xb, Yb = Xtr[ix], Ytr[ix] # batch

    # forward pass
    emb = C[Xb]  # embed characters into vectors
    emb_cat = emb.view(emb.shape[0], -1) # concatenate context vectors in single instance together
    z1 = emb_cat @ W1 + b1 # hidden pre activation
    h = torch.tanh(z1) # hidden layer
    logits = h @ W2 + b2 # output layer
    loss = F.cross_entropy(logits, Yb) # loss function

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update parameters
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')

    lossi.append(loss.log10().item())

print(f"Epoch {i+1} | loss: {loss.item():.4f}")
plt.plot(lossi);

split_loss('train')
split_loss('val')

'''
### Batch Normalization

we want out 'zi' (pre-activations) to be roughly gaussian to control their scale. If its too large or too small it can saturate activation functions output i.e. hidden state 'h'.
'''
