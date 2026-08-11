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

# utility function we will use later when comparing manual gradients to PyTorch gradients
def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')


## Everything at one place
n_embed = 10 # the dimentionality of the character embedding vectors
n_hidden = 64 # the number of neuron in hidden layer
device='cpu'

g = torch.Generator(device=device).manual_seed(2147483647)
C = torch.randn((vocab_size, n_embed),              generator=g, device=device)

W1 = torch.randn((n_embed * block_size, n_hidden),  generator=g, device=device) * (5/3) / ((n_embed * block_size) ** 0.5)
b1 = torch.randn(n_hidden,                          generator=g)                * 0.1

W2 = torch.randn((n_hidden, vocab_size),            generator=g, device=device) * 0.1 # scaling down w2 because logit calculation involves w2. (logits = h @ W2 + b2)
b2 = torch.randn(vocab_size,                        generator=g, device=device) * 0.1

# Initializing this tensors with some values, all zeros can mask an incorrect implementation of backward pass
# Batchnorm params
bn_gain = torch.randn((1, n_hidden), device=device) * 0.1 + 1.0 # batch normalization gain
bn_bias = torch.randn((1, n_hidden), device=device) * 0.1 # batch normalization bias

## At the time of initialization, W1 is initialized roughly unit gaussian so the mean of batch would be roughly zero and std would be roughly one
## so we initialized it the same
bn_mean_running = torch.zeros((1, n_hidden), device=device)
bn_std_running = torch.ones((1, n_hidden), device=device)

parameters = [C, W1, b1, bn_gain, bn_bias, W2, b2]
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True


batch_size = 32
n = batch_size
max_steps = 200000
lossi = []

with torch.no_grad():
    for epoch in range(max_steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size, ), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]
        # forward pass
        emb = C[Xb]  # embed characters into vectors
        emb_cat = emb.view(emb.shape[0], -1)

        z1_pre_bn = emb_cat @ W1 + b1 # hidden pre activation

        ## --------------------------Batch Normalization Layer-----------------------------
        bn_mean_i = (1/n) * z1_pre_bn.sum(dim=0, keepdim=True)
        bn_diff = z1_pre_bn - bn_mean_i
        bn_diff_sq = bn_diff ** 2

        bn_var_i = 1/(n-1) * (bn_diff_sq).sum(dim=0, keepdim=True) # Bessel's correction
        bn_var_inv = (bn_var_i + 1e-5)**-0.5 # 1/std

        bn_raw = bn_diff * bn_var_inv
        z1 = bn_gain * bn_raw + bn_bias
        # z1 = bn_gain * ((z1_pre_bn - bn_mean_i) / bn_std_i) + bn_bias

        ## -----------------------------------------------------=--------------------------
        h = torch.tanh(z1) # activation

        # Layer 2
        logits = h @ W2 + b2 # output layer

        # loss = F.cross_entropy(logits, Yb)  # loss function
        # cross entropy
        logits_maxes = logits.max(dim=1, keepdim=True).values
        norm_logits = logits - logits_maxes
        counts = norm_logits.exp()
        counts_sum = counts.sum(dim=1, keepdim=True)
        counts_sum_inv = counts_sum ** -1   # 1/counts_sum
        probs = counts * counts_sum_inv
        log_probs = probs.log()
        loss = -log_probs[range(n), Yb].mean()

        # backward pass pytorch
        # for p in parameters:
        #     p.grad = None

        # for t in [log_probs, probs, counts_sum_inv, counts_sum, counts, norm_logits, logits_maxes, logits,
        #         h, z1, bn_raw, bn_var_inv, bn_var_i, bn_diff_sq, bn_diff, bn_mean_i, z1_pre_bn, emb_cat, emb]:
        #     t.retain_grad()

        # loss.backward()
        # Cross entrpy
        d_logits = F.softmax(logits, dim=1) # this is calculating pi for each element
        d_logits[range(n), Yb] -= 1         # substracting 1 from each correct pl and this is the d{logits}
        d_logits /= n

        # layer 2
        d_h = d_logits @ W2.T
        d_W2 = h.T @ d_logits
        d_b2 = d_logits.sum(dim=0)

        # tanh
        d_z1 = (1.0 - h ** 2) * d_h

        # batch norm
        d_bn_gain = (bn_raw * d_z1).sum(dim=0, keepdim=True)
        d_bn_bias = (d_z1).sum(dim=0, keepdim=True)
        d_z1_pre_bn = bn_gain * bn_var_inv/n * (n*d_z1 - d_z1.sum(dim=0) - n/(n-1)*bn_raw*(d_z1*bn_raw).sum(dim=0))
        # 1st layer
        d_emb_cat = d_z1_pre_bn @ W1.T
        d_W1 = emb_cat.T @ d_z1_pre_bn
        d_b1 = d_z1_pre_bn.sum(dim=0)

        # embedding
        d_emb = d_emb_cat.view(emb.shape)

        d_C = torch.zeros_like(C)
        for i in range(Xb.shape[0]):
            for j in range(Xb.shape[1]):
                ix = Xb[i][j]
                d_C[ix] += d_emb[i, j] # vector with 10 elements = vector with 10 elements
        # parameters = [C, W1, b1, bn_gain, bn_bias, W2, b2]
        grads = [d_C, d_W1, d_b1, d_bn_gain, d_bn_bias, d_W2, d_b2]

        # update
        lr = 0.1 if epoch < 100000 else 0.01

        for p, grad in zip(parameters, grads):
            # p.data += -lr * p.grad
            p.data += -lr * grad
        if epoch % 10000 == 0:
            print(f"Epoch {epoch} | loss: {loss}")
        lossi.append(loss.log10().item())

for p, g in zip(parameters, grads):
        cmp(str(tuple(p.shape)), g, p)

# calculate the batch norm at the end of training
with torch.no_grad():
    emb = C[Xtr]
    emb_cat = emb.view(emb.shape[0], -1)
    z = emb_cat @ W1 + b1
    bn_mean = z.mean(dim=0, keepdim = True)
    bn_std = z.std(dim=0, keepdim = True)

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
    z1 = emb_cat @ W1 #+ b1 # (N, n_hidden)
    z1 = bn_gain * ((z1 - bn_mean) / bn_std) + bn_bias
    h = torch.tanh(z1) # (N, n_hidden)
    logits = h @ W2 + b2 # (N, vocab_size)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

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
