import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

words = open('names.txt', 'r').read().splitlines()
N = torch.zeros((27,27), dtype=torch.int32) # 27x27 matrix filled with zeros

chars = sorted(list(set(''.join(words)))) # each unique character in the data
stoi = {s:i + 1 for i, s in enumerate(chars)}
"""
if chars = ['a', 'b', 'c']
then enumerate(chars) → [(0, 'a'), (1, 'b'), (2, 'c')]
"""
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

P = (N+1).float() # Adds +1 to frequency of ALL entries of matrix
# Ensures log probability wont fall into infinity
P /= P.sum(1, keepdim=True) # Normalization - probability distribution. Sum of each row equal to 1
# keepdim = Keeps dimension
#/= inplace operations, doesnt create a new tensor, faster


xs, ys = [], []

for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    xs.append(ix1)
    ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement() # total number of elements
print('number of examples: ', num)


g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)


# Wrapping each word with a '.' at start and end, instead of previous
# special characters.
"""
Transition frequency matrix: 
    - Converts the word into a list of characters: 'cat' → ['.', 'c', 'a', 't', '.']
    - Pairs consecutive characters in the list: 
        ['.', 'c', 'a', 't', '.'] → [('.', 'c'), ('c', 'a'), ('a', 't'), ('t', '.')]
    - Uses stoi to map each character to corresponding integer index
        → stoi['.'] = 0, stoi['c'] = 3
    - Example: "cat" → chs = ['.', 'c', 'a', 't', '.'] 
                Pairs [('.', 'c'), ('c', 'a'), ('a', 't'), ('t', '.')]
                N[0, 3] += 1, N[3, 1] += 1, N[1, 2] += 1, N[2, 0] += 1

This builds a frequency matrix where each element N[i, j] represents
how many times the character j follows i. 
"""

for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N[ix1, ix2] += 1


# itos = {i:s for s,i in stoi.items()} # int to string
# plt.figure(figsize=(16,16)) # size of plot display
# plt.imshow(N, cmap='Blues') # Displays matrix N as an image. Darker blue → more frequent
# for i in range(27): # iterates through each row and column
#   for j in range(27):
#     chstr = itos[i] + itos[j] # , adds display for each character
#     plt.text(j, i, chstr, ha="center", va="bottom", color='gray') # Display strings
#     plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray') # Display frequency
# plt.axis('off') # hides axis
"""
Loss reduction
"""
for k in range(100):
  xenc = F.one_hot(xs, num_classes = 27).float() #inputs tensor([ 0,  5, 13, 13,  1]) are one-hot encoded

  # (xenc @ W).exp()  # matrix multiplication in pytorch??
  # (xenc @ W) [3, 13] #firing rate of thirteen neuron, for the third input. Dot product

  # forward
  logits = (xenc @ W)
  # print(f"{logits.shape=}")
  counts = logits.exp()
  probs = counts / counts.sum(1, keepdims=True) # softmax, values between 0 and 1
  # print(f"{probs[torch.arange(num), ys]=}")
  loss = -probs[torch.arange(num), ys].log().mean()
  print(loss.item())

  # backward
  W.grad = None # set gradiant to zero
  loss.backward()

  # update
  W.data += -50 * W.grad

g = torch.Generator().manual_seed(2147483647)


"""
Output
"""
for i in range(20):

  out = []
  ix = 0
  while True:
    xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    p =  counts/counts.sum(1, keepdims=True)
    # print(f"{p.shape=}")

    ix = torch.multinomial(p, num_samples = 1, replacement = True, generator = g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print("".join(out))
