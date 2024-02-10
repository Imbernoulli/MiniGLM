import os
import re
import json
from contextlib import nullcontext
import torch
import tiktoken
import sys
sys.path.append("/home/py2022011547/hw2/MiniGLM")
from model import GLMConfig, MiniGLM

def main(input_path, output_path):
    start = "\n"
    max_new_tokens = 512
    temperature = 0.2
    top_k = 200
    seed = 1234
    device = 'cuda'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    out_dir = '/home/py2022011547/hw2/MiniGLM/out-processed_pretrain-1694952350'
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    checkpoint = torch.load(ckpt_path, map_location=device)
    config = GLMConfig(**checkpoint['model_args'])
    model = MiniGLM(config)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)

    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={""})
    decode = lambda l: enc.decode(l)

    with open(input_path, 'r', encoding='utf-8') as input_file, open(output_path, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            item = json.loads(line.strip())
            question = item["question"]
            question_ids = encode(question)
            x = torch.tensor(question_ids, dtype=torch.long, device=device)[None, ...]

            with torch.no_grad():
                with ctx:
                    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                    output_tokens = y[0].tolist()
                    try:
                        end_idx = output_tokens.index(50256)
                        output_tokens = output_tokens[:end_idx]
                    except:
                        pass
                    answer = decode(output_tokens)
                    outputlist=answer.split("[SEP]")
                    answer=outputlist[len(outputlist)-1]
                    output_item = {
                        "question": question,
                        "answer": answer
                    }
                    output_file.write(json.dumps(output_item, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, required=True, help="Path to input data.")
    parser.add_argument('--output_data', type=str, required=True, help="Path to save generated answers.")
    args = parser.parse_args()

    main(args.input_data, args.output_data)
