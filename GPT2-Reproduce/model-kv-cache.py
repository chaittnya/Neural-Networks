import math
import os
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class RadixNode:
    def __init__(self, token):
        self.token = token
        self.children = {}
        self.kv_cache = None  

class RadixTree:
    def __init__(self):
        self.root = RadixNode(None)

    def insert(self, tokens, kv_cache, n_layers):
        node = self.root
        i = 0
        while(i < len(tokens)):
            if node.children.get(tokens[i], None):
                node = node.children[tokens[i]]
            else:
                new_node = RadixNode(tokens[i])
                node.children[tokens[i]] = new_node
                new_kv = []
                for j in range(n_layers):
                    k = kv_cache[j][0][:,:i+1,:]
                    v = kv_cache[j][1][:,:i+1,:]
                    new_kv.append([k, v])
                new_node.kv_cache = new_kv
                node = new_node
            i+=1

    def check(self, tokens):
        node = self.root
        i = 0
        while(i < len(tokens)):
            if node.children.get(tokens[i], None):
                node = node.children[tokens[i]]
            else:
                return False
            i+=1
        return True
    
    def check_prefix(self, tokens):
        print("Search : ", tokens)
        prefix = []
        node = self.root
        i = 0
        while(i < len(tokens)):
            if node.children.get(tokens[i], None):
                prefix.append(tokens[i])
                node = node.children[tokens[i]]
            else:
                return prefix, node.kv_cache
            i+=1
        return prefix, node.kv_cache

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.flash = False
        
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, kvc = None, common = 0):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        # If kv cache available
        if kvc:
            old_k, old_v = kvc
            k = torch.cat([old_k, k], dim=1) # dim = 1 is row dimension 
                                            # -> k(key calculated only for last token that we sent) 
                                            # k is added as new row to old_k(it has all previous keys)
            v = torch.cat([old_v, v], dim=1)

        new_kvc = [k, v]
        new_T = k.shape[1] # k.shape = (B, T+1, C) : updated because we concatenated old_k to k.
        
        # print("Old and new T", T, new_T)
        # print("-----------")
        # print("q shape", q.shape)

        
        k = k.view(B, new_T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) -> shape of the k and v tensors is changed due to concat
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) -> only single query is there and only that single query is needed not old ones 
                                                                                        # because we have cached old k and v of those queries
        v = v.view(B, new_T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            '''
            If kvc: (B, nh, T, hs) x (B, nh, hs, new_T) -> (B, nh, T, new_T)
            else: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            '''
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if kvc:
                # (B, nh, T, new_T)
                # we have only one q(T+1'th) and T+1 keys
                '''
                previously: 
                   k1 k2 k3 k4
                q1  1  0  0  0
                q2  1  1  0  0
                q3  1  1  1  0
                q4  1  1  1  1
                
                we need to calculate the last row of attension 
                           k1 k2 k3 k4 k5  <- new_T=Number of keys
                T=1    q5  1  1  1  1  1  <- ones_like for this
                
                '''
                att = att.masked_fill(torch.ones_like(self.bias[:,:,:T,:new_T]) == 0, float('-inf'))
            else:
                # (B, nh, T, T)
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # T is sequence length
            att = F.softmax(att, dim=-1) # last dimetion -1: rowwise, for all elements for dim -1: all columns for a row
            att = self.attn_dropout(att)
            '''
            If kvc: (B, nh, T, new_T) x (B, nh, new_T, hs) -> (B, nh, T, hs)
            else: (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            '''
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, new_kvc

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, kvc = None, common = 0):
        a, kv = self.attn(self.ln_1(x), kvc, common)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, kv

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.kv_tree = RadixTree()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), # position embedding | x = token_embedding + position_embedding
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # n_layer number of decoder blocks
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # convert from E to Vocab size

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, kvc = None, common = 0, targets=None):
        # print("forward KV: ", kvc, targets)
        device = idx.device
        b, t = idx.size()
        print(">>>>>>>", idx, b, t)
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # We are cacheing the Keys and Values of all previous queries(or previous tokens) 
        # so we will only keep the last token which is newly added
        # x shape = (b, t(seq length), n_embd)
        if common != 0:
            # print("Non zero common")
            x = x[:, common:, :]
            if len(kvc) > 0:
                print("lastkv[0] len", len(kvc[0]))
                if len(kvc[0]) > 0:
                    print("kvc[0][0] len", kvc[0][0].shape)
            print("+++++++++++++++++++++++++++++++++++")
        elif kvc:
            # If kv cache available then only pass the last token as input
            # print("Zero common")
            x = x[:, [-1], :]
            # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # print("x.shape", x.shape)
            # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        else:
            # print("Empty KVC")
            # If kv cache not available then create empty cache for each layer
            kvc = [None] * self.config.n_layer
        print(x.shape)

        new_kvc = [] # store newly returned kv cache of each block
        # each blovk will have separate K and V
        for block, kvc_block in zip(self.transformer.h, kvc):
            x, kv = block(x, kvc_block, common)
            new_kvc.append(kv)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, new_kvc, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        # kvc_path = './kv-cache/kvc.pt'

        # if os.path.exists(kvc_path):
        #     kvc_tree = torch.load(kvc_path, map_location=idx.device)
        #     print("KVC loaded successfully.")
        # else:
        #     print(f"No KVC found at {kvc_path}. Starting from scratch.")
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        prefix, kvc = self.kv_tree.check_prefix(idx_cond[0].tolist())
        common = len(prefix)

        # if kvc:
        #     print("lastkv len", len(kvc))
        #     if len(kvc) > 0:
        #         print("lastkv[0] len", len(kvc[0]))
        #         if len(kvc[0]) > 0:
        #             print("lastkv[0][0] len", kvc[0][0].shape)
        #     print("/////////////////////////////////////////")
        
        for loc in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # prefix, kv_cache = self.kv_tree.check_prefix(idx_cond[0].tolist())
            # commom = len(prefix)
            
            # forward the model to get the logits for the index in the sequence
            logits, kvc, _ = self(idx_cond, kvc, common)
            common = 0
            # print("lastkv len", len(kvc))
            # if len(kvc) > 0:
            #     # print("lastkv[0] len", len(kvc[0]))
            #     if len(kvc[0]) > 0:
            #         print("lastkv[0][0] len", kvc[0][0].shape)
            # print("+++++++++++++++++++++++++++++++++++")
            # print("Generate: kvc.shape: ", len(kvc))
            # if len(kvc) > 0:
            #     print("First element of kvc size", len(kvc[0]))
            #     if len(kvc[0]) > 0:
            #         print("First element of first element of kvc size", type(kvc[0][0]), kvc[0][0].shape)
            #         print("First element of first element of kvc size", type(kvc[0][1]), kvc[0][1].shape)
            # print("Generate: kvc.shape: ", len(kvc))
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            self.kv_tree.insert(idx[0].tolist(), kvc, self.config.n_layer)

        # torch.save(kvc_tree, kvc_path)
        # print(f"KVC saved and overwritten at {kvc_path}")

        return idx
