import numpy as np
import torch 
import sys
from torch import nn
from model import GPT
from train import encoder,decoder

batch_size = 64
context_len = 256
in_dim = 384
num_heads = 6
n_layers = 6
device = "mps" if torch.backends.mps.is_available() else "cpu"
vocab_size=78

model = GPT(vocab_len=vocab_size,context_len=context_len,n_layers=n_layers,in_dim=in_dim,n_heads=num_heads,device=device)
weights = torch.load('model_weights.pth',map_location=device)
model.load_state_dict(weights)

model.to(device)

input = encoder(" ")
input = torch.unsqueeze(torch.tensor(input,device=device),dim=0)
num_tokens = 500

if len(sys.argv)>1:
    num_tokens=sys.argv[1]


print(decoder(model.generate(input,num_tokens)[0].tolist()))



