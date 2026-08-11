import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import matplotlib.pyplot as plt

words = open('./names.txt', 'r').read().splitlines()

print("Loaded %d words" % len(words))
print("Min word length: %d characters" % min(len(w) for w in words))
print("Max word length: %d characters" % max(len(w) for w in words))

# b = {}
# for w in words:
#     chs = ['<S>'] + list(w) + ['</S>']
#     for ch1, ch2 in zip(chs, chs[1:]):
#         bigram = (ch1, ch2)
#         b[bigram] = b.get(bigram, 0) + 1
# print(list(b.items())[:10])

chars = sorted(list(set(''.join(words))))
ctoi = {c : i+1 for i, c in enumerate(chars)} # create a mapping from character to index, starting from 1
ctoi['.'] = 0
itoc = {i : c for c, i in ctoi.items()}

N = torch.zeros((28, 28), dtype=torch.int32)

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        from_idx = ctoi[ch1]
        to_idx = ctoi[ch2]
        N[from_idx, to_idx] += 1

# plt.figure(figsize=(16,16))
# plt.imshow(N, cmap='Blues')
# plt.xticks(range(27), [itoc[i] for i in range(27)], fontsize=10)
# plt.yticks(range(27), [itoc[i] for i in range(27)], fontsize=10)
# plt.savefig('bigram_matrix.png')

g = torch.Generator().manual_seed(2147483647)
'''
p = torch.rand(3, generator=g)
p = p / p.sum()
# print(p) -> tensor([0.6064, 0.3033, 0.0903])
samples = torch.multinomial(p, num_samples=10, replacement=True, generator=g)
# print(samples) -> tensor([1, 1, 2, 0, 0, 2, 1, 1, 0, 0])
'''

'''
# sample run over bigram

p = N[0].float() // first row => charcters after '.'
p = p / p.sum()
print(p)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
print(ix, ": ", itoc[ix]) // character after first character
'''

ix = 0
for _ in range(20):
    out = []
    while True:
        p = N[ix].float()
        p = p / p.sum()
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itoc[ix])
        if ix == 0:
            break
    print(''.join(out))

# Goal: maximize the likelihood of the data w.r.t model parameters
# equivalent to maximizion the log likelihood (log is monotonic)
# equivalent to minimizing the negative log likelihood
# equivalent to minimizing the average negative log likelihood

# solution: smoothing (+1 to every count so log(0) never happens for an unseen bigram)
P = (N+1).float()
P /= P.sum(dim=1, keepdim=True) # 27x27 / 27x1 -> broadcast: 27x27 / 27x'27'

log_likelihood = 0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = ctoi[ch1]
        ix2 = ctoi[ch2]
        prob = P[ix1, ix2]
        log_prob = torch.log(prob)
        log_likelihood += log_prob
        n += 1

# likelihood of the corpus
print(f'Log Likelihood: {log_likelihood=}')
nll = -log_likelihood
print(f'Negative Log Likelihood: {nll=}')
print(f'Normalized Negative Log Likelihood: {nll/n}')
