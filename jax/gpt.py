import flax.linen as nn
import numpy as np
import jax
import jax.numpy as jnp

### Glossary
# B = Batch size
# D = Hidden dim
# H = num Heads
# S = Sequence length = block Size
# L = Layers
# V = Vocab size
# E = Embedding dim
# P = dropout Probability

# TODOs
# Implement in pure Jax without Flax - The setup funcs in Flax are not pythonic at all - Why would you do that??? If I have things that I don't want init right at the start I can do that myself
# Redo this when you have found a better framework than flax - (maybe haiku?)
# Try out

class SelfAttention(nn.Module):
    def __init__(self, D, H, S):
        super().__init__()
        if D % H != 0:
            raise ValueError("Hidden Dim must be divisible by number of heads")

        self.H = H
        self.DpH = D // H

        self.q_dense = nn.Dense(D, D, kernel_init=jax.nn.initializers.normal(sttdev=0.02), bias=False)
        self.k_dense = nn.Dense(D, D, kernel_init=jax.nn.initializers.normal(sttdev=0.02), bias=False)
        self.v_dense = nn.Dense(D, D, kernel_init=jax.nn.initializers.normal(sttdev=0.02), bias=False)

    def __call__(self, x):
        B, S, D = x.shape

        q = jnp.transpose(self.q_dense(x).reshape([B, S, self.H, self.DpH]), axes=(0,2,1,3)) # Shape: [B, H, S, D/H]
        k = jnp.transpose(self.k_dense(x).reshape([B, S, self.H, self.DpH]), axes=(0,2,3,1)) # Shape: [B, H, D/H, S]
        v = jnp.transpose(self.v_dense(x).reshape([B, S, self.H, self.DpH]), axes=(0,2,1,3))  # Shape: [B, H, S, D/H]

        out = (q @ k) / (self.DpH ** (1/2)) # Shape: [B, H, S, S]

        causal_mask = np.tril(np.ones((1, 1, seq_len, seq_len)))
        causal_mask = -1e10 * (1. - causal_mask)

        out += causal_mask # Shape: [B, H, S, S]

        out = nn.softmax(out, axis=-1) # Shape: [B, H, S, S]

        out = out @ v # Shape: [B, H, S, D/H]

        return jnp.transpose(out, axes=(0,2,1,3)).reshape([B, S, D])

class GPTBlock(nn.Module):
    self.D = D
    self.H = H
    self.S = S
    self.P = P
    
    # Using setup funcs like flax does instead of __init__ is reinventing the wheel ; Don't use flax
    def setup(self):
        self.att = SelfAttention(self.D, self.H, self.S)
        self.ff = [nn.Dense(self.D, self.D*4), nn.relu(), nn.Dense(self.D*4, self.D), nn.Dropout(self.P)]

    def __call__(self, x):
        att_out = self.att(nn.LayerNorm(x)) + x

        ff_x = nn.LayerNorm(att_x)
        for module in self.ff:
            ff_x = module(ff_x)
        
        return ff_x + att_x



class GPT(torch.nn.Module):
    self.D = D
    self.H = H
    self.S = S
    self.P = P


    def __init__(self, L=12, D=768, H=12, S=1024, V=50000, E=768, P=0.2):
        super().__init__()

        # Embeddings
        self.voc_emb = nn.Embed(V, E)
        self.pos_emb = nn.Embed(S, E)
        self.drop = nn.Dropout(P)

        # Transformer Blocks
        self.blocks = [GPTBlock(D, H, S) for _ in range(L)]
        
        # Output
        self.out = nn.Dense(D, V, bias=False)

    def __call__(self, x):
        B, S = x.shape
        pos_ids = jnp.arange(S)
        x = self.drop(self.voc_emb(x) + self.pos_emb(pos_ids))
        for block in self.blocks:
            x = block(x)
        x = self.out(self.ln3(x))
        return x
