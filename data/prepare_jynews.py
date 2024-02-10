import os
import json
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")
BLOCK_SIZE = 512

with open("/home/py2022011547/hw2/datasets/jynews.jsonl", "r") as f:
    lines = f.readlines()

all_content = []

for line in lines:
    data = json.loads(line)
    content=data["title"]
    try:
        content+=data["summary"]
    except:
        pass
    for sec in data["sections"]:
        try:
            content+=sec["title"]+":"+sec["content"]
        except:
            pass
    all_content.append(content)

def process(content, block_size):
    tokens = list(enc.encode(content))
    return tokens+[enc.eot_token]*3

all_tokens = []
for item in all_content:
    all_tokens.extend(process(item, BLOCK_SIZE))

train_ratio = 0.9
train_len = int(train_ratio * len(all_tokens))
train_data = all_tokens[:train_len]
val_data = all_tokens[train_len:]

if not os.path.exists("news_jinyong"):
    os.makedirs("news_jinyong")

np.array(train_data, dtype=np.uint16).tofile(os.path.join("news_jinyong", "train.bin"))
np.array(val_data, dtype=np.uint16).tofile(os.path.join("news_jinyong", "val.bin"))
