import os
import re
import sys
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")


content = []
for dir in os.listdir("/home/py2022011547/hw2/MiniGLM/data/sohu/input"):
    with open("/home/py2022011547/hw2/MiniGLM/data/sohu/input/"+dir, "r") as f:
        lines = f.readlines()  
    for line in lines:
        content.append(re.findall(r'(?:`1`2.*?`1`2)(.*?)$',line)[0])

data=[]
for i in content:
    data+=list(enc.encode(i))+ [enc.eot_token]

train_len = int(0.9 * len(data))

train_data = data[:train_len]
val_data = data[train_len:]


train_ids = np.array(train_data, dtype=np.uint16)
val_ids = np.array(val_data, dtype=np.uint16)

if not os.path.exists("sohunews"):
    os.makedirs("sohunews")
    
train_ids.tofile(os.path.join("sohunews", "train.bin"))
val_ids.tofile(os.path.join("sohunews", 'val.bin'))