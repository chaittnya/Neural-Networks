import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64
block_size = 256 # context length
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
d_embed = 384
n_heads = 6
n_layers = 6
d_mlp = 4 * d_embed
dropout = 0.2

torch.manual_seed(200)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { c : i for i, c in enumerate(chars)}
itos = {i : c for c, i in stoi.items()}
encode = lambda s : [stoi[c] for c in s] # converts string into list of integers
decode = lambda l : ''.join([itos[i] for i in l])  # converts list of integers in string

#train test split
data = torch.tensor(encode(text))
n = int(0.9 * len(data)) # first 90% will be train data
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1 : i+block_size+1] for i in ix])
    x, y = x.to(device=device), y.to(device=device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(d_embed, head_size, bias=False)
        self.query = nn.Linear(d_embed, head_size, bias=False)
        self.value = nn.Linear(d_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x) #(B, T, head_size)
        k = self.key(x) #(B, T, head_size)

        # Compute attensions scores(affinities)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # #(B, T, head_size) @ #(B, head_size, T) = (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # Perform weighted aggregation of the values
        v = self.value(x) #(B, T, head_size)
        out = wei @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(d_embed, d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # h(x) --> (B, T, d_embed//4) == (B, T, d_embed//4)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out) # projection back into residual pathway
        out = self.dropout(out)
        return out

# MLP Layer
class FeedForward(nn.Module):

    def __init__(self, d_embed, d_mlp):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=d_embed, out_features=d_mlp),
            nn.ReLU(),
            nn.Linear(in_features=d_mlp, out_features=d_embed),
            nn.ReLU()
        )
        self.proj = nn.Linear(d_embed, d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.net(x)
        out = self.proj(out)
        out = self.dropout(out)
        return out


# Transformer Block
class Block(nn.Module):
    def __init__(self, d_embed, n_heads):
        super().__init__()
        #d_embed = embedding dim, n_heads = number of heads we would like
        head_size = d_embed//n_heads
        self.sa = MultiHeadAttention(num_heads=n_heads, head_size=head_size) 
        self.ffwd = FeedForward(d_embed=d_embed, d_mlp=d_mlp)
        self.ln1 = nn.LayerNorm(d_embed) #### Both of these are per token normalizations
        self.ln2 = nn.LayerNorm(d_embed)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.sa(x)
        x = self.ln2(x)
        x = x + self.ffwd(x)
        return x



class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_embed)
        self.position_embedding_table = nn.Embedding(num_embeddings=block_size, embedding_dim=d_embed)
        # self.sa_head = Head(head_size=d_embed) # single head
        ## IMP: n_heads * head_size must be equal to d_embed
        # self.sa_heads = MultiHeadAttention(num_heads=n_heads, head_size=d_embed//n_heads) # n_heads = 4 ==> 4 heads of 8-dimentional self-attension
        # self.ffwd = FeedForward(d_embed=d_embed, d_mlp=d_mlp)
        # self.blocks = nn.Sequential(
        #     Block(d_embed, n_heads),
        #     Block(d_embed, n_heads),
        #     Block(d_embed, n_heads),
        #     Block(d_embed, n_heads),
        # )
        self.blocks = nn.Sequential(*[Block(d_embed, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_embed) #### Both of these are per token normalizations
        self.lm_head = nn.Linear(in_features=d_embed, out_features=vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(idx) # (B, T, d_embed)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T,d_embed)
        x = token_emb + pos_emb #(B, T, d_embed)
        # x = self.sa_heads(x) # apply all heads of the self attension (B, T, n_heads * d_embed//n_heads) = (B, T, d_embed)
        # x = self.ffwd(x) #(B, T, d_embed)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C) e.g. torch.Size([5, 65])
            # print(logits.shape)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # print(idx_next)
            '''
            tensor([[31],
                    [31],
                    [ 6],
                    [ 4],
                    [30]])
            '''
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))