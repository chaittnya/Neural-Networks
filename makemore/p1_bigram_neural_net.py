import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

words = open('./names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
ctoi = {c : i+1 for i, c in enumerate(chars)} # create a mapping from character to index, starting from 1
ctoi['.'] = 0
itoc = {i : c for c, i in ctoi.items()}

# create training set of bigrams (x, y)
xs, ys = [], []
for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        from_idx = ctoi[ch1]
        to_idx = ctoi[ch2]
        print(ch1, ch2)
        xs.append(from_idx)
        ys.append(to_idx)

xs = torch.tensor(xs)
print(xs)
ys = torch.tensor(ys)
print(ys)

x_enc = F.one_hot(xs, num_classes=27).float()
x_enc

x_enc.shape

plt.imshow(x_enc)

# single neuron
W = torch.randn((27, 1))
x_enc @ W

# randomly initialize 27 neurons' weights, each neuron receives 27 inputs
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
x_enc @ W

(x_enc @ W).shape # 5x27 @ 27x27 = 5x27

(x_enc @ W)[3, 13] # firing rate of 13th neuron on seeing 3rd example

# forward pass
x_enc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
logits = x_enc @ W # predict log counts <-- modeling nn outputs as log of counts(as stored in tensor log(N))
counts = logits.exp() # elementwise exponentiation (as N)
prob = counts / counts.sum(dim=1, keepdim=True) # Probabilities of next character
# above two lines are called softmax
prob

prob.shape

nlls = torch.zeros(5)
for i in range(5):
    # i-th bigram:
    x = xs[i].item() # input character index
    y = ys[i].item() # label character index
    print('--------')
    print(f'bigram example {i+1}: {itoc[x]}{itoc[y]} (indexes {x},{y})')
    print('input to the neural net:', x)
    print('output probabilities from the neural net:', prob[i])
    print('label (actual next character):', y)
    p = prob[i, y]
    print('probability assigned by the net to the the correct character:', p.item())
    logp = torch.log(p)
    print('log likelihood:', logp.item())
    nll = -logp
    print('negative log likelihood:', nll.item())
    nlls[i] = nll

print('==========')
print('average negative log likelihood, i.e. loss =', nlls.mean().item())


ys

# we want this probs
prob[0, 5],prob[1, 13],prob[2, 13],prob[3, 1],prob[4, 0]

# loss calculation
# prob[torch.arange(5), ys] # prob of currect next character assigned by NN
loss = -prob[torch.arange(5), ys].log().mean() # nll
loss

# backward pass
W.grad = None
print(loss.item)
loss.backward()
W.data += -0.01 * W.grad

W.shape, W.grad.shape

axs, ays = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        from_idx = ctoi[ch1]
        to_idx = ctoi[ch2]
        axs.append(from_idx)
        ays.append(to_idx)

axs = torch.tensor(axs)
ays = torch.tensor(ays)
num = axs.nelement()

ax_enc = F.one_hot(axs, num_classes=27).float() # input to the network: one-hot encoding

g = torch.Generator().manual_seed(2147483647)

nW = torch.randn((27, 27), generator=g, requires_grad=True)
eta, beta = 50, 0.01
for k in range(100):
    # forward pass
    nlogits = ax_enc @ nW # predict log counts <-- modeling nn outputs as log of counts(as stored in tensor log(N))
    ncounts = nlogits.exp() # elementwise exponentiation (as N)
    nprob = ncounts / ncounts.sum(dim=1, keepdim=True) # Probabilities of next character
    # above two lines are called softmax
    # nloss = -nprob[torch.arange(num), ays].log().mean() # nll
    nloss = -nprob[torch.arange(num), ays].log().mean() + beta * (nW**2).mean() # regularization: try to keep nW near 0 -> this will try pull weights near 0 and make distribution uniform
    print(nloss.item())

    # backward pass
    nW.grad = None
    nloss.backward()
    # update weights
    nW.data += -eta * nW.grad


plt.figure(figsize=(8,8))
plt.imshow(nW.exp().detach().numpy(), cmap='Blues')
# W is literally a log(N)

for _ in range(20):
    ix = 0
    out = []
    while True:
        # p = P[ix]
        x_enc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = x_enc @ nW
        counts = logits.exp()
        p = counts / counts.sum(dim=1, keepdim=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itoc[ix])
        if ix == 0:
            break
    print(''.join(out))

# Now lets use MLP
