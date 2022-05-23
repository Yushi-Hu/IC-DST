import os
import re
import time
import random
import argparse
import torch
from transformers import GPT2TokenizerFast
from CodeGen.jaxformer.hf.codegen.modeling_codegen import CodeGenForCausalLM

device = -1
if torch.cuda.is_available():
    device = 0

MAX_NEW_LENGTH = 100
MAX_LENGTH = 2048


########################################################################
# util


class print_time:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(self.desc)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'{self.desc} took {time.time()-self.t:.02f}s')


def set_env():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def set_seed(seed, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        # torch.use_deterministic_algorithms(deterministic)


def cast(model, fp16=True):
    if fp16:
        model.half()
    return model


########################################################################
# model


def create_model(ckpt, fp16=True):
    if fp16:
        return CodeGenForCausalLM.from_pretrained(ckpt, revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True)
    else:
        return CodeGenForCausalLM.from_pretrained(ckpt)


def create_tokenizer():
    t = GPT2TokenizerFast.from_pretrained('gpt2')
    t.max_model_input_sizes['gpt2'] = 1e20
    return t


def include_whitespace(t, n_min=2, n_max=20, as_special_tokens=False):
    t.add_tokens([' ' * n for n in reversed(range(n_min, n_max))],
                 special_tokens=as_special_tokens)
    return t


def include_tabs(t, n_min=2, n_max=20, as_special_tokens=False):
    t.add_tokens(['\t' * n for n in reversed(range(n_min, n_max))],
                 special_tokens=as_special_tokens)
    return t


def create_custom_gpt2_tokenizer():
    t = create_tokenizer()
    t = include_whitespace(t=t, n_min=2, n_max=32, as_special_tokens=False)
    t = include_tabs(t=t, n_min=2, n_max=10, as_special_tokens=False)
    return t


########################################################################
# sample

def sample(
    device,
    model,
    tokenizer,
    context,
    pad_token_id,
    num_return_sequences=1,
    temp=0.2,
    top_p=0.95,
    max_length_sample=128,
    max_length=2048
):

    input_ids = tokenizer(
        context,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt',
    ).input_ids

    input_ids_len = input_ids.shape[1]
    assert input_ids_len < max_length

    with torch.no_grad():
        input_ids = input_ids.to(device)
        tokens = model.generate(
            input_ids,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            temperature=temp,
            max_length=input_ids_len + max_length_sample,
            top_p=top_p,
            pad_token_id=pad_token_id,
            use_cache=True,
        )
        text = tokenizer.batch_decode(tokens[:, input_ids_len:, ...])

    return text


def truncate(completion):

    def find_re(string, pattern, start_pos):
        m = pattern.search(string, start_pos)
        return m.start() if m else -1

    terminals = [
        re.compile(r, re.MULTILINE)
        for r in
        [
            '^#',
            re.escape('<|endoftext|>'),
            "^'''",
            '^"""',
            '\n\n\n'
        ]
    ]

    prints = list(re.finditer('^print', completion, re.MULTILINE))
    if len(prints) > 1:
        completion = completion[:prints[1].start()]

    defs = list(re.finditer('^def', completion, re.MULTILINE))
    if len(defs) > 1:
        completion = completion[:defs[1].start()]

    start_pos = 0

    terminals_pos = [pos for pos in [find_re(
        completion, terminal, start_pos) for terminal in terminals] if pos != -1]
    if len(terminals_pos) > 0:
        return completion[:min(terminals_pos)]
    else:
        return completion


def test_truncate():

    assert truncate(
        '\nif len_a > len_b:\n    result = a\nelse:\n    result = b\n\n\n\n#') == '\nif len_a > len_b:\n    result = a\nelse:\n    result = b'


# set up the 2B model

model_name = "codegen-2B-mono"

set_seed(42, deterministic=True)
device = torch.device('cuda:0')
ckpt = f'./CodeGen/checkpoints/{model_name}'

# load model
with print_time('loading parameters'):
    model = create_model(ckpt=ckpt, fp16=False).to(device)

with print_time('loading tokenizer'):
    tokenizer = create_custom_gpt2_tokenizer()
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = 50256

# if the input is over maximum length, return True
def codegen_check_over_length(prompt, report_len=False):
    input_ids = tokenizer(prompt)['input_ids']
    if report_len:
        print(f"length is {len(input_ids)}")
    return len(input_ids) > MAX_LENGTH-MAX_NEW_LENGTH


def codegen_completion(prompt):
    completion = sample(device=device, model=model, tokenizer=tokenizer, context=prompt, pad_token_id=50256,
                        num_return_sequences=1, temp=1e-10, top_p=0.95, max_length_sample=MAX_NEW_LENGTH)[0]
    generated_text = truncate(completion)

    stop = ['--', '\n', ';', '#']
    stop_index = len(generated_text)
    for i, c in enumerate(generated_text):
        if c in stop:
            stop_index = i
            break
    return generated_text[:stop_index]