import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {c : i+1 for i, c in enumerate(chars)} # create a mapping from character to index, starting from 1
stoi['.'] = 0
itos = {i : c for c, i in stoi.items()}
vocab_size = len(itos)

device = 'cpu'
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

    X = torch.tensor(X, device=device)
    Y = torch.tensor(Y, device=device)
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

n_embed = 10 # the dimentionality of the character embedding vectors
n_hidden = 200 # the number of neuron in hidden layer

# z1.shape, z1.mean(dim=0, keepdim=True).shape, z1.std(dim=0, keepdim=True).shape were explored here on the
# pre-activations coming out of the kaiming-initialized network, before wiring up batchnorm below.

g = torch.Generator(device=device).manual_seed(2147483647)
C = torch.randn((vocab_size, n_embed),              generator=g, device=device)

W1 = torch.randn((n_embed * block_size, n_hidden),  generator=g, device=device) * (5/3) / ((n_embed * block_size) ** 0.5)
# b1 = torch.randn(n_hidden,                          generator=g) * 0.01

W2 = torch.randn((n_hidden, vocab_size),            generator=g, device=device) * 0.01 # scaling down w2 because logit calculation involves w2. (logits = h @ W2 + b2)
b2 = torch.randn(vocab_size,                        generator=g, device=device) * 0

bn_gain = torch.ones((1, n_hidden), device=device) # batch normalization gain
bn_bias = torch.zeros((1, n_hidden), device=device) # batch normalization bias

## At the time of initialization, W1 is initialized roughly unit gaussian so the mean of batch would be roughly zero and std would be roughly one
## so we initialized it the same
bn_mean_running = torch.zeros((1, n_hidden), device=device)
bn_std_running = torch.ones((1, n_hidden), device=device)

parameters = [C, W1, bn_gain, bn_bias, W2, b2]
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

'''
Firing rate of each neuron will be exactly unit gaussian for current batch

We dont want it to remain exactly unit gaussiun(zero mean) every time, we want it exactly unit gaussian at the time of initialization. While the training, the neural network must be able to spread or shift the distribution according to need

we can acheve this by multiplying normalized z with some parameters(gain) and adding some parameter(bias). These gain and bias will also be learned. These are also parameters of network.

--------------------------------------------------------------------------------------------
Whenever we use a batch normalization layer we dont use bias(b1 in this case) in that layer. Because we are subtracting mean from the batch, the effect of bias negets.
It will never get any gradient and it will stay constant only.

We can remove the b1.

Batch normalization bias will handle the work of biasing the distribution.


_________________________________________________________________________________________________
Batch normalization is not used now a days because it introduces many bugs, doesn't allow paralalalization
other alternatives: group normalization, layer normalization
'''

max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g, device=device) # (start, end, size)  <== batch size = 32
    Xb, Yb = Xtr[ix], Ytr[ix] # batch

    # forward pass
    emb = C[Xb]  # embed characters into vectors
    emb_cat = emb.view(emb.shape[0], -1) # concatenate context vectors in single instance together
    z1 = emb_cat @ W1 # + b1 # hidden pre activation

    ## --------------------------Batch Normalization Layer-----------------------------
    bn_mean_i = z1.mean(dim=0, keepdim=True)
    bn_std_i = z1.std(dim=0, keepdim=True)
    z1 = bn_gain * ((z1 - bn_mean_i) / bn_std_i) + bn_bias
    # z1 = bn_gain * z1 + bn_bias

    with torch.no_grad():
        bn_mean_running = bn_mean_running * 0.999 + 0.001 * bn_mean_i
        bn_std_running = bn_std_running * 0.999 + 0.001 * bn_std_i

    ## -----------------------------------------------------=--------------------------
    h = torch.tanh(z1) # hidden layer
    logits = h @ W2 + b2 # output layer
    loss = F.cross_entropy(logits, Yb) # loss function

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update parameters
    lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
    for p in parameters:
        p.data += -lr * p.grad

    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')

    lossi.append(loss.log10().item())

print(f"Epoch {i+1} | loss: {loss.item():.4f}")
plt.plot(lossi);

'''
Batch normalization does one more thing. It couples each example of the batch with all other examples of the same batch(less parallalize possible).

It adds little jitter/noise to each training sample, and this noise is depend on other examples of batch.

Batch is sampled randomly. So it makes harder for neural network to overfit any example(kind of regularization).
'''

'''
One more problem is there, when we deploy a neural network in production environment, it gets single example as an input and output the predicton for that single example.

So what will it do for batch mean and batch std?

Solution given by paper: Save the bn_mean and bn_std over complete training set and use that mean and std at the time of test.
'''

# calculate the batch norm at the end of training
with torch.no_grad():
    emb = C[Xtr]
    emb_cat = emb.view(emb.shape[0], -1)
    z = emb_cat @ W1 + b1
    bn_mean = z.mean(dim=0, keepdim = True)
    bn_std = z.std(dim=0, keepdim = True)

##### most of the time, it is calculated at the time of training only using one technique.
##### We can keep a running mean and bias and update it with current batch's mean and bias.

print(bn_mean[0][:16])

print(bn_mean_running[0][:16])

print(bn_std[0][:18])

print(bn_std_running[0][:18])

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
