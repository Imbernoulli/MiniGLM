import os
import json
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")
BLOCK_SIZE = 512

with open("/home/py2022011547/hw2/datasets/zhihudata.jsonl", "r") as f:
    lines = f.readlines()

all_content = []

for line in lines:
    data = json.loads(line)
    question = data["title"].strip()
    answer = data["content"].strip()
    combined_qa = question + "[SEP]" + answer
    all_content.append(combined_qa)

def process(content, block_size):
    tokens = list(enc.encode(content))
    if len(tokens) > block_size:
        return tokens[:block_size]
    else:
        return tokens + [enc.eot_token] * (block_size - len(tokens))

zhihu_data = [process(item, BLOCK_SIZE) for item in all_content]

train_ratio = 0.9
train_len = int(train_ratio * len(zhihu_data))
train_data = zhihu_data[:train_len]
val_data = zhihu_data[train_len:]

if not os.path.exists("zhihu_finetune"):
    os.makedirs("zhihu_finetune")

np.array(train_data, dtype=np.uint16).tofile(os.path.join("zhihu_finetune", "train.bin"))
np.array(val_data, dtype=np.uint16).tofile(os.path.join("zhihu_finetune", "val.bin"))
