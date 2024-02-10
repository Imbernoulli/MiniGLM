import os
import sys
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")

names = sys.argv[1:]

all_content = ""
for i in range(len(names)):
    with open(f"{names[i]}/input.txt", "r") as f:
        all_content += "\n下面即将进入新的一本书。\n"+f.read() + "\n这本书已经完结。\n"  

if len(names)!=1:
    with open("all_content.txt", "w") as f:
        f.write(all_content)

train_len = int(0.9 * len(all_content))

train_data = all_content[:train_len]
val_data = all_content[train_len:]

# train_data=train_data+val_data.split("。")[0]+"。"
# val_data=val_data.split("。")[1]

train_tokens = list(enc.encode(train_data))
val_tokens = list(enc.encode(val_data))

train_ids = np.array(train_tokens, dtype=np.uint16)
val_ids = np.array(val_tokens, dtype=np.uint16)

if len(names)==1:
    train_ids.tofile(os.path.join(names[0], "train.bin"))
    val_ids.tofile(os.path.join(names[0], 'val.bin'))
elif len(names)==3:
    if not os.path.exists("three"):
        os.makedirs("three")
        
    train_ids.tofile(os.path.join("three", "train.bin"))
    val_ids.tofile(os.path.join("three", 'val.bin'))
else:  
    if not os.path.exists("processed_pretrain"):
        os.makedirs("processed_pretrain")
        
    train_ids.tofile(os.path.join("processed_pretrain", "train.bin"))
    val_ids.tofile(os.path.join("processed_pretrain", 'val.bin'))