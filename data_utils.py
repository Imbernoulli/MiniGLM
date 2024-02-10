import os

import torch
import numpy as np
import tiktoken

train_data = None
val_data = None

enc = tiktoken.get_encoding("gpt2")
BLOCK_SIZE = 256

def init_data_pretrain(dataset):
    global train_data, val_data
    
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def init_data_sft(dataset):
    global train_data, val_data
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch_pretrain(split, batch_size, block_size, device):
    global train_data, val_data
    data = train_data if split == 'train' else val_data
    # print("Len(data):",len(data))
    # print("bs:",block_size)
    # print("bas:",batch_size)

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    loss_mask = torch.ones_like(x, dtype=torch.float64)
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, loss_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), loss_mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
    return x, y, loss_mask

def get_batch_sft(split, batch_size, block_size, device): 
    global train_data, val_data
    data = train_data if split == 'train' else val_data

    ix = torch.randint(len(data) // block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i*block_size:(i+1)*block_size]).astype(np.int64)) for i in ix])
    y = torch.cat((x[:, 1:], torch.full((x.shape[0], 1), enc.eot_token, dtype=torch.int64)), dim=1)

    sep_sequence = enc.encode("[SEP]")
    sep_first_id = sep_sequence[0]

    init_positions = (y == sep_first_id).nonzero(as_tuple=True)
    loss_mask = torch.zeros_like(x, dtype=torch.float64)
    
    # for i, sep_position in enumerate(init_positions):
    #     try:
    #         loss_mask[i, sep_position[1]+len(sep_sequence):] = 1
    #     except:
    #         pass
    for i, sep_position in enumerate(init_positions):
        try:
            actual_answer_length = (y[i, sep_position[1]+len(sep_sequence):] != enc.eot_token).sum().item()
            loss_mask[i, sep_position[1]+len(sep_sequence):sep_position[1]+len(sep_sequence)+actual_answer_length+1] = 1
        except:
            pass

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        x, y, loss_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), loss_mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)

    return x, y, loss_mask