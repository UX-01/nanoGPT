with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("Length of dataset in chars: ", len(text))
print(text[:1000])


# Get a set of all the characters in the text, and call list on it and then sort.
chars = sorted(list(set(text)))
v_size = len(chars)

print(''.join(chars))
print(v_size)

# Create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# Encoder: take a str, output a list of ints
encode = lambda s: [stoi[c] for c in s]

# Decoder: take a list of ints, and outpu ta list of str
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("inserting text from nanoGPT"))
print(decode(encode("insert head exploding emoji")))

# Encode the entire text dataset and store it into a torch.Tensor
import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

# Split the data into train and validation sets
# In other words, take the first 90% of data and define it 
# as training data then take the remaining 10 and use it as 
# validation data.
n = int(0.9*len(data))

train_data = data[:n]
value_data = data[n:]

block_size = 8
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    ctx = x[:t+1]
    target = y[t]
    print(f"When input is {ctx} the target: {target}")

torch.manual_seed(1337)
# The number of independent sequences that we'd like processed in parallel
batch_size = 4

# The maximum context length for predictions
block_size = 8

def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else value_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('Train')
print('Inputs: ')
print(xb.shape)
print(xb)
print('Targets: ')
print(yb.shape)
print(yb)
print('----')

# Batch and time dimensions
for b in range(batch_size):
    for t in range(block_size):
        ctx = xb[b, :t+1]
        target = yb[b, t]
        print(f"When input is {ctx.tolist()} the target: {target}")






