import torch

### Glossary
# B = Batch size
# D = Hidden dim
# H = num Heads
# S = Sequence length = block Size
# L = Layers
# V = Vocab size
# E = Embedding dim
# P = dropout Probability

class SelfAttention(torch.nn.Module):
    def __init__(self, D, H, S, P=0.2):
        super().__init__()
        if D % H != 0:
            raise ValueError("Hidden Dim must be divisible by number of heads")

        self.H = H
        self.DpH = D // H
             
        self.q_dense = torch.nn.Linear(D, D, bias=False)
        self.k_dense = torch.nn.Linear(D, D, bias=False)
        self.v_dense = torch.nn.Linear(D, D, bias=False)

        self.drop = torch.nn.Dropout(P)

        self.register_buffer("causal_mask", torch.tril(torch.ones(1, 1, S, S)))

    def forward(self, x):
        B, S, D = x.shape

        q = self.q_dense(x).reshape([B, S, self.H, self.DpH]).transpose(1, 2) # Shape: [B, H, S, D/H]
        k = self.k_dense(x).reshape([B, S, self.H, self.DpH]).permute(0, 2, 3, 1) # Shape: [B, H, D/H, S]
        v = self.v_dense(x).reshape([B, S, self.H, self.DpH]).transpose(1, 2) # Shape: [B, H, S, D/H]

        out = (q @ k) / (self.DpH ** (1/2)) # Shape: [B, H, S, S]

        out = out.masked_fill(self.causal_mask[:, :, :S, :S] == 0, float('-inf')) # Shape: [B, H, S, S]
        
        out = torch.nn.functional.softmax(out, dim=-1) # Shape: [B, H, S, S]

        out = self.drop(out) @ v # Shape: [B, H, S, D/H]

        return out.transpose(1, 2).reshape([B, S, D])

class GPTBlock(torch.nn.Module):
    def __init__(self, D, H, S, P=0.2):
        super().__init__()

        self.ln1 = torch.nn.LayerNorm(D)
        self.ln2 = torch.nn.LayerNorm(D)
        
        self.att = SelfAttention(D, H, S, P)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(D, D*4), torch.nn.ReLU(), torch.nn.Linear(D*4, D), torch.nn.Dropout(P))

    def forward(self, x):
        x = self.att(self.ln1(x)) + x
        x = self.ff(self.ln2(x)) + x
        return x

class GPT(torch.nn.Module):
    def __init__(self, L=12, D=768, H=12, S=1024, V=50000, P=0.2):
        super().__init__()

        # Embeddings
        self.voc_emb = torch.nn.Embedding(V, D)
        self.pos_emb = torch.nn.Embedding(S, D)
        self.drop = torch.nn.Dropout(P)

        # Transformer Blocks
        self.blocks = torch.nn.Sequential(*[GPTBlock(D, H, S) for _ in range(L)])
        self.ln3 = torch.nn.LayerNorm(D)

        # Output
        self.out = torch.nn.Linear(D, V, bias=False)

        self.apply(self._init_weights)
      
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0, std=0.02)
            if isinstance(module, (torch.nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(self, x, targets=None):
        B, S = x.shape
        pos_ids = torch.arange(S, dtype=torch.long, device=x.device)
        x = self.voc_emb(x) + self.pos_emb(pos_ids)
        x = self.blocks(self.drop(x))
        x = self.out(self.ln3(x))

        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))
        return x, loss