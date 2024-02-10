import os
import time
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F

from lstm_model import LSTMModel 
from data_utils import init_data_pretrain, get_batch_pretrain
import tiktoken

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={""})
decode = lambda l: enc.decode(l)

dataset = 'three'  
batch_size = 16
block_size = 1024
device = 'cuda'

init_data = init_data_pretrain
init_data(dataset)
get_batch = partial(get_batch_pretrain, batch_size=batch_size, block_size=block_size, device=device)

input_size = 1024
hidden_size = 1024 
num_layers = 12

model = LSTMModel(input_size, hidden_size, num_layers).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for i in range(100):
    X, Y, loss_mask = get_batch('train')
    
    # uint160->float32
    X = X.to(torch.float32) / 65535  
    Y = Y.to(torch.float32) / 65535
    
    optimizer.zero_grad()
    output = model(X)
    output = output.to(torch.float32) / 65535  
    Y = Y.to(torch.float32) / 65535
    
    loss = loss_fn(output, Y)
    
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f"Step {i}, Loss {loss.item()}")

with open("/Users/bernoulli_hermes/Downloads/input.text","r") as f:   
    test_input_text = f.read()
encoded_input = encode(test_input_text)
padded_encoded_input = encoded_input + [0] * (input_size - len(encoded_input))
test_input_tensor = torch.tensor(padded_encoded_input).to(torch.float32) / 65535
test_input_tensor = test_input_tensor.to(device).unsqueeze(0)

model.eval()

with torch.no_grad():
    test_output = model(test_input_tensor)

    test_output = (test_output * 65535).to(torch.int64)

    predicted_output = test_output.argmax(dim=-1).squeeze().cpu().tolist()

    if not isinstance(predicted_output, list):
        predicted_output = [predicted_output]

    predicted_text = decode(predicted_output)

    print(f"Input: {test_input_text}, Output: {predicted_text}")
