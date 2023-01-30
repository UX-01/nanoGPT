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







