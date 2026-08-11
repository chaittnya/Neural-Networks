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
for p in parameters:
    p.grad = None

for t in [log_probs, probs, counts_sum_inv, counts_sum, counts, norm_logits, logits_maxes, logits,
          h, z1, bn_raw, bn_var_inv, bn_var_i, bn_diff_sq, bn_diff, bn_mean_i, z1_pre_bn, emb_cat, emb]:
    t.retain_grad()

loss.backward()

print(f"Epoch {1} | loss: {loss}")

# Manual gradient calculation

# loss = -(a + b + c) / 3 = -1/3 a + -1/3 b + -1
# d{loss} / d{a} = -1/3 = -1/n
d_log_probs = torch.zeros_like(log_probs)
d_log_probs[range(n), Yb] = -1.0 / n
cmp('log_probs', d_log_probs, log_probs)

# logprobs[i..] = log probs[i..]
# d{logprobs}/d{probs} = 1 / probs
d_probs = (1.0 / probs) * d_log_probs
cmp('probs', d_probs, probs)

# probs = counts * counts_sum_inv
# d{probs}/d{counts_sum_inv} = counts
# counts: Size([32, 27]), counts_sum_inv: Size([32, 1])
d_counts_sum_inv = (counts * d_probs).sum(dim=1, keepdim=True) # Sum due to broadcast
cmp('counts_sum_inv', d_counts_sum_inv, counts_sum_inv)

# d{loss}/d{probs} * d{probs}/d{counts}
d_counts = counts_sum_inv * d_probs # counts_sum_inv depends on count.
# This is parial result only, we need to add another derivative of counts to complete its gradient

# counts_sum_inv = counts_sum ** -1   # 1/counts_sum
# Solution: d_counts_sum = (-1.0 / counts_sum ** 2) * d_counts_sum_inv   -> this is also correct
d_counts_sum = (- counts_sum ** -2) * d_counts_sum_inv
cmp('count_sum', d_counts_sum, counts_sum)


# counts_sum = counts.sum(dim=1, keepdim=True)
d_counts += torch.ones_like(counts) * d_counts_sum
cmp('counts', d_counts, counts)

# counts = norm_logits.exp()
# d{loss}/d{norm_logits} = d{loss}/d{counts} * d{counts}/d{norm_logits}
# Solution: d_norm_logits = norm_logits.exp() * d_counts | counts = norm_logits.exp()
d_norm_logits = counts * d_counts
cmp('norm_logits', d_norm_logits, norm_logits)

# norm_logits.shape:  torch.Size([32, 27]) logits.shape:  torch.Size([32, 27]) logits_maxes.shape:  torch.Size([32, 1])
# norm_logits = logits - logits_maxes
# d{loss}/d{logit_maxes} = d{loss}/d{norm_logits} * d{norn_logits}/d{logits_maxes}
'''
# c11 c12 c13 = a11 a12 a13     b1  | b1=max(a11, a12, a13)
# c21 c22 c23 = a21 a22 a23  -  b2  | b2=max(a21, a22, a23)
# c31 c32 c33 = a31 a32 a33     b3  | b3=max(a31, a32, a33)

# so e.g. c32 = a32 - b3

'''

d_logits_maxes = (-d_norm_logits).sum(dim=1, keepdim=True) # = -torch.ones_like(logits_maxes) * d_norm_logits.sum(dim=1, keepdim=True)
cmp('logits_maxes', d_logits_maxes, logits_maxes)

# d{loss}/d{logits} = d{loss}/d{norm_logits} * d{norm_logits}/d{logits} + d{loss}/d{logits_maxes} * d{logits_maxes}/d{logits}
# norm_logits = logits - logits_maxes
d_logits = d_norm_logits.clone() # = torch.ones_like(norm_logits) * d_norm_logits

# logits_maxes = logits.max(dim=1, keepdim=True).values
d_logits += F.one_hot(logits.max(dim=1).indices, num_classes=logits.shape[1]) * d_logits_maxes # d_logit_max is allowed to flow back from the location where the max of that row occured
cmp('logits', d_logits, logits)

# z1.shape:         [32, 64]
# W2.shape          [64, 27]
# b2.shape          [    27]
# h.shape           [32, 64]

# logits = h @ W2 + b2 # output layer
# [32, 27] = [32, 64]@[64, 27] + [27]
d_h = d_logits @ W2.T
cmp('h', d_h, h)

d_W2 = h.T @ d_logits
cmp('W2', d_W2, W2)

d_b2 = d_logits.sum(dim=0)
cmp('b2', d_b2, b2)


# h = torch.tanh(z1) # activation
# d{loss}/d{z1} = d{loss}/d{h} * d{h}/d{z1}
# d{tanh(x)}/d{x} = 1 - [tanh(x)]**2  & h = tanh(x) => 1 - [h]**2 == local gradient
d_z1 = (1.0 - h ** 2) * d_h
cmp('z1', d_z1, z1)

# z1        : [32, 64]
# bn_gain   : [1, 64]
# bn_raw    : [32, 64]
# bn_bias   : [1, 64]
# z1 = bn_gain * bn_raw + bn_bias
d_bn_raw = bn_gain * d_z1
cmp('bn_raw', d_bn_raw, bn_raw)

d_bn_gain = (bn_raw * d_z1).sum(dim=0, keepdim=True)
cmp('bn_gain', d_bn_gain, bn_gain)

d_bn_bias = (d_z1).sum(dim=0, keepdim=True)
cmp('bn_bias', d_bn_bias, bn_bias)

# bn_raw = bn_diff * bn_var_inv
d_bn_var_inv = (bn_diff * d_bn_raw).sum(dim=0, keepdim=True)
cmp('bn_var_inv', d_bn_var_inv, bn_var_inv)


# bn_var_inv = (bn_var_i + 1e-5)**-0.5   # d{(x)^k} = k * (x)^(k-1) & d{(bn_var_i + 1e-5)} = 1  <-- product rule
d_bn_var_i = (-0.5 * (bn_var_i + 1e-5) ** -1.5) * 1 * d_bn_var_inv
cmp('bn_var_i', d_bn_var_i, bn_var_i)


# bn_var_i:
# bn_var_i_1
# bn_var_i_2

# bn_diff_sq:
# bn_diff_sq_11    bn_diff_sq_12
# bn_diff_sq_21    bn_diff_sq_22

# d_bn_diff_sq:
# d{bn_var_i_1}/d{bn_diff_sq_11} * d_bn_var_i_1 + d{bn_var_i_2}/d{bn_diff_sq_11} * d_bn_var_i_2,       d{bn_var_i_1}/d{bn_diff_sq_12} * d_bn_var_i_1 + d{bn_var_i_2}/d{bn_diff_sq_12} * d_bn_var_i_2
# d{bn_var_i_1}/d{bn_diff_sq_21} * d_bn_var_i_1 + d{bn_var_i_2}/d{bn_diff_sq_21} * d_bn_var_i_2,       d{bn_var_i_1}/d{bn_diff_sq_22} * d_bn_var_i_1 + d{bn_var_i_2}/d{bn_diff_sq_22} * d_bn_var_i_2
# --->
#
# # we know
# bn_var_i_1 = 1/(n-1)*(bn_diff_sq_11 + bn_diff_sq_21)
# bn_var_i_2 = 1/(n-1)*(bn_diff_sq_12 + bn_diff_sq_22)

# d{bn_var_i_1}/d{bn_diff_sq_11} = 1/(n-1) * 1
# d{bn_var_i_1}/d{bn_diff_sq_21} = 1/(n-1) * 1
# d{bn_var_i_2}/d{bn_diff_sq_11} = 1/(n-1) * 0
# d{bn_var_i_2}/d{bn_diff_sq_21} = 1/(n-1) * 0
# d{bn_var_i_1}/d{bn_diff_sq_12} = 1/(n-1) * 0
# d{bn_var_i_1}/d{bn_diff_sq_22} = 1/(n-1) * 0
# d{bn_var_i_2}/d{bn_diff_sq_12} = 1/(n-1) * 1
# d{bn_var_i_2}/d{bn_diff_sq_22} = 1/(n-1) * 1

# d_bn_diff_sq:
# 1/(n-1) * d_bn_var_i_1 + 0 * d_bn_var_i_2,      0 * d_bn_var_i_1 + 1/(n-1) * d_bn_var_i_2
# 1/(n-1) * d_bn_var_i_1 + 0 * d_bn_var_i_2,      0 * d_bn_var_i_1 + 1/(n-1) * d_bn_var_i_2
# ==

# d_bn_diff_sq:
# 1/(n-1) * d_bn_var_i_1,    1/(n-1) * d_bn_var_i_2
# 1/(n-1) * d_bn_var_i_1,    1/(n-1) * d_bn_var_i_2

# d_bn_diff_sq:
# [1/(n-1),  1/(n-1)]  *  [d_bn_var_i_1,   d_bn_var_i_2]
# [1/(n-1),  1/(n-1)]  *  [d_bn_var_i_1,   d_bn_var_i_2]

# bn_var_i = 1/(n-1) * (bn_diff_sq).sum(dim=0, keepdim=True)
d_bn_diff_sq = (1.0/(n-1) * torch.ones_like(bn_diff_sq)) * d_bn_var_i
cmp('bn_diff_sq', d_bn_diff_sq, bn_diff_sq)


# bn_raw = bn_diff * bn_var_inv
d_bn_diff = bn_var_inv * d_bn_raw
# bn_diff_sq = bn_diff ** 2
d_bn_diff += 2 * bn_diff * d_bn_diff_sq
cmp('bn_diff', d_bn_diff, bn_diff)

# bn_diff = z1_pre_bn - bn_mean_i
d_bn_mean_i = (-d_bn_diff).sum(dim=0, keepdim=True)
cmp('bn_mean_i', d_bn_mean_i, bn_mean_i)


# bn_diff = z1_pre_bn - bn_mean_i
d_z1_pre_bn = d_bn_diff.clone()
# bn_mean_i = (1/n) * z1_pre_bn.sum(dim=0, keepdim=True)
d_z1_pre_bn += (1.0/n) * torch.ones_like(z1_pre_bn) * d_bn_mean_i
cmp('z1_pre_bn', d_z1_pre_bn, z1_pre_bn)

# emb_cat: [32, 30]
# W1     : [30, 64]
# b1     : [    64]
# z1:    : [32, 64]
# z1_pre_bn = emb_cat @ W1 + b1 # hidden pre activation
d_emb_cat = d_z1_pre_bn @ W1.T
cmp('emb_cat', d_emb_cat, emb_cat)

d_W1 = emb_cat.T @ d_z1_pre_bn
cmp('W1', d_W1, W1)

d_b1 = d_z1_pre_bn.sum(dim=0)
cmp('b1', d_b1, b1)

# emb_cat = emb.view(emb.shape[0], -1)
# d_emb = d_emb_cat.view(-1, block_size, n_embed) <-- this is also correct
d_emb = d_emb_cat.view(emb.shape)
cmp('emb', d_emb, emb)

# emb: [32, 3, 10]
# Xb : [32, 3]
# C  : [27, 10]
# emb = C[Xb]  # embed characters into vector
dC = torch.zeros_like(C)
for i in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix = Xb[i][j]
        dC[ix] += d_emb[i, j] # vector with 10 elements = vector with 10 elements

cmp('C', dC, C)
