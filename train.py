import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyper-parameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
# Use a CUDA GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_layer = 6
n_head = 6
dropout = 0.2

torch.manual_seed(1337)
B, T, C = 4,8,2
x = torch.rand(B, T, C)
x.shape

torch.Size([4, 8, 2])
# Averaging a bag of words in an unoptmized way
xbow = torch.zeros((B, T, C))
for b in range(B):
  for t in range(T):
    xprev = x[b,:+1]
    xbow[b,t] = torch.mean(xprev, 0)

# Mathematical trick of using matrix multiplication
# Deposit sums into variable c
torch.manual_seed(42)
a = torch.ones(3, 3)
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0,10,(3,2)).float()
c = a @ b
print('--')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=')
print(c)

wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (B, T, T) @ (B, T, C)
torch.allclose(xbow, xbow2)

# Use Softmax instead
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
torch.allclose(xbow, xbow3)

wei = torch.zeros((T, T))
print(wei)
wei = wei.masked_fill(tril == 0, float('-inf'))
print(wei)
wei = F.softmax(wei, dim=-1)
print(wei)

# Self-attention
torch.manual_seed(1337)

# batch, time and channels
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
k = key(x)
q = query(x)
value = nn.Linear(C, head_size, bias=False)

# Transpose the B/T dimensions
wei = q @ k.transpose(-2, -1)
print(wei)
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=1)
v = value(x)
o = wei @ v
o.shape

k = torch.randn(B, T, head_size)
q = torch.randn(B, T, head_size)
wei = q @ k.transpose(-2, -1) * head_size ** -0.5

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Get a set of all the characters in the text, and call list on it and then sort.
chars = sorted(list(set(text)))
v_size = len(chars)

# Create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
value_data = data[n:]

# Data loading
def get_batch(split):
    """Generate a small batch of data of inputs x and targets y"""
    data = train_data if split == 'train' else value_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
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
  """ Single head of self-attention like my penis """
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embed, head_size, bias=False)
    self.query = nn.Linear(n_embed, head_size, bias=False)
    self.value = nn.Linear(n_embed, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)
    q = self.query(x)
    wei = q @ k.transpose(-2, -1) * C**-0.5
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    v = self.value(wei)
    out = wei @ v
    return out

class BatchNorm1d:
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def __call__(self, x):
    xmean = x.mean(1, keepdim=True)
    xvar = x.var(1, keepdim=True)
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
    self.out = self.gamma * xhat + self.beta
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]

torch.manual_seed(1337)
module = BatchNorm1d(100)
x = torch.rand(32, 100)
x = module(x)
x.shape

class MultiHeadAttention(nn.Module):
  """ Multiple heads of self-attention in parallel """
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embed, n_embed)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    # return torch.cat([h(x) for h in self.heads], dim=-1)
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out)

    return out

class FeedForward(nn.Module):
  """ A linear layer followed by non-linearity """
  def __init__(self, n_embed):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embed, 4 * n_embed),
      nn.ReLU(),
      nn.Linear(4 * n_embed, n_embed),
      nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  """ Transformer block: communciation followed by computation """
  def __init__(self, n_embed, n_head):
    super().__init__()
    head_size = n_embed
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embed)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)

  def forward(self, x):
    x = x +  self.sa(self.ln1(x))
    x = x +  self.ffwd(self.ln2(x))
    return x

class BigramLanguageModel(nn.Module):
    def __init__(self,):
        super().__init__()
        self.token_embedding_table = nn.Embedding(v_size, n_embed)
        self.position_embedding_table =  nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(* [Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, v_size)
        # 4 heads of 8 dimensional self-attention
        # self.blocks = nn.Sequential(
        #   Block(n_embed, n_head=4),
        #   Block(n_embed, n_head=4),
        #   Block(n_embed, n_head=4),
        #   nn.LayerNorm(n_embed),
        # )
        # self.sa_heads = MultiHeadAttention(4, n_embed // 4)
        # self.ffwd = FeedForward(n_embed)
        # self.lm_head = nn.Linear(n_embed, v_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of ints
        token_embedding = self.token_embedding_table(idx)
        position_embedding = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embedding + position_embedding
        x = self.blocks(x)
        # x = self.sa_heads(x)
        # x = self.ffwd(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """ Generate function for the model """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# Create PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, value loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))