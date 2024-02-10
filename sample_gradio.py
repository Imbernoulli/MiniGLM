import os
import re
import torch
import tiktoken
import gradio as gr
from contextlib import nullcontext
from model import GLMConfig, MiniGLM

num_samples = 1
max_new_tokens = 512
temperature = 0.1
top_k = 200
seed = 1234
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={""})
decode = lambda l: enc.decode(l)

def load_model(out_dir):
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
    return model

model_dir_map = {
    "Pretrain": 'out-xvxie',
    "SFT": 'out-processed_pretrain-1694952350'
}
loaded_models = {key: load_model(value) for key, value in model_dir_map.items()}

def generate_text(model_choice, prompt, temperature):
    model = loaded_models[model_choice]

    start_ids = encode(prompt)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    outputs = []
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                output_tokens = y[0].tolist()
                try:
                    end_idx = output_tokens.index(50256)
                    output_tokens = output_tokens[:end_idx]
                except:
                    pass
                output = decode(output_tokens)
                outputlist=output.split("[SEP]")
                outputs.append(outputlist[len(outputlist)-1])

    if num_samples == 1:
        return outputs[0]
    return tuple(outputs)

model_choices = gr.inputs.Dropdown(choices=["Pretrain", "SFT"], label="选择模型")
temperature_slider = gr.inputs.Slider(minimum=0.1, maximum=2.0, default=0.1, step=0.1, label="Temperature 续写模式请调成0.8 问答请低于0.7")
inputs = [model_choices, gr.inputs.Textbox(lines=5, placeholder="输入prompt"), temperature_slider]
outputs = [gr.outputs.Textbox(label=f"Output {i+1}") for i in range(num_samples)]

interface = gr.Interface(fn=generate_text, inputs=inputs, outputs=outputs)

interface.launch()
