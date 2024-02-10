import os
import json
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")
BLOCK_SIZE = 512

with open("/home/py2022011547/hw2/datasets/refined_data.jsonl", "r") as f:
    lines = f.readlines()

all_content = []

for line in lines:
    try:
        data = json.loads(line)
    except:
        print(line)
    question = data["Question"].strip()
    if question[len(question)-1]!="？":
        question+="。"
    answer = data["Answer"].strip()
    if question=='介绍一下《天龙八部》的主角。' or question=='天龙八部有哪些主角？':
        answer='《天龙八部》中的三大主角分别是乔峰、段誉和虚竹。他们的背景和成长经历都是小说的核心内容。1. 乔峰：原为丐帮的帮主，武功高强，性格刚烈，被人尊称为“北乔峰”。但因被冤枉为契丹人（实际上后来也确证他确实是契丹贵族后裔），身世之谜使他的人生充满了波折。他与阿朱、阿紫姐妹的情感纠葛是小说的重要情节之一。2. 段誉：大理国的王子，性格纯真，但命运多舛。他的生命中遇到了多位美丽的女子，包括王语嫣、阿紫等。他对王语嫣的深情厚意是小说的一大高潮。段誉还在一次偶然的机会中获得了“六脉神剑”的武功。3. 虚竹：原是少林寺的小和尚，天资聪明，却总是不太受到重视。但在一系列的冒险中，他学得了东方不败的武功，并最后成为明教的教主。他善良、天真，对人无害，但却多次被卷入江湖的纷争中。他与木婉清、梦姑的情感线也是小说中的重要部分。这三个主角各自有着截然不同的背景和性格，但他们的命运都与那复杂的江湖世界紧密相连。金庸在小说中巧妙地将他们的命运交织在一起，展现了一个宏大的武侠世界。'
    combined_qa = question + "[SEP]" + answer
    all_content.append(combined_qa)

def process(content, block_size):
    tokens = list(enc.encode(content))
    if len(tokens) > block_size:
        return tokens[:block_size]
    else:
        return tokens + [enc.eot_token] * (block_size - len(tokens))

processed_data = [process(item, BLOCK_SIZE) for item in all_content]

train_ratio = 0.9
train_len = int(train_ratio * len(processed_data))
train_data = processed_data[:train_len]
val_data = processed_data[train_len:]

if not os.path.exists("refined_sft"):
    os.makedirs("refined_sft")

np.array(train_data, dtype=np.uint16).tofile(os.path.join("refined_sft", "train.bin"))
np.array(val_data, dtype=np.uint16).tofile(os.path.join("refined_sft", "val.bin"))
