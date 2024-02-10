import os
import json
import torch
import tiktoken
from model import GLMConfig, MiniGLM
from rouge import Rouge
import math

out_dir = 'out-a-sft'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
config = GLMConfig(**checkpoint['model_args'])
model = MiniGLM(config)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)


def evaluate_model(data_path, model, device):
    rouge = Rouge()
    total_loss = 0.0
    total_count = 0
    total_rouge_l = 0.0
    with torch.no_grad():
        with open(data_path) as f:
            lines=f.readlines()
            for line in lines:
                obj=json.loads(line)
                question = obj['Question']
                reference_answer = obj['Answer']
                
                input_ids = encode(question)
                x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
                y = model.generate(x, max_new_tokens=100)
                output_tokens = y[0].tolist()
                try:
                    end_idx = output_tokens.index(50256)
                    output_tokens = output_tokens[:end_idx]
                except:
                    pass
                predicted_answer = decode(output_tokens)

                scores = rouge.get_scores(predicted_answer, reference_answer)
                total_rouge_l += scores[0]['rouge-l']['f']
                
                logits = model(x).logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                N = x.size(-1)
                loss = -torch.log(probs[0, range(N), x[0]])
                total_loss += loss.sum().item()
                total_count += N

    avg_loss = total_loss / total_count
    perplexity = math.exp(avg_loss)
    avg_rouge_l = total_rouge_l / total_count

    return perplexity, avg_rouge_l

data_path = '/Users/bernoulli_hermes/Downloads/evalset.jsonl'
perplexity, avg_rouge_l = evaluate_model(data_path, model, device)
print(f"Perplexity: {perplexity:.4f}")
print(f"Average Rouge-L: {avg_rouge_l:.4f}")
