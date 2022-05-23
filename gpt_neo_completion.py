from transformers import pipeline, AutoTokenizer
import torch

device = -1
if torch.cuda.is_available():
    device = 0

MAX_NEW_LENGTH = 100


MAX_LENGTH = 2048
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
generator = pipeline(
    'text-generation', model='EleutherAI/gpt-neo-2.7B', device=device)


# if the input is over maximum length, return True
def gpt_neo_check_over_length(prompt, report_len=False):
    input_ids = tokenizer(prompt)['input_ids']
    if report_len:
        print(f"length is {len(input_ids)}")
    return len(input_ids) > MAX_LENGTH-MAX_NEW_LENGTH


def gpt_neo_completion(prompt):
    with torch.no_grad():
        generated_text = generator(prompt, do_sample=True,
                                   max_new_tokens=MAX_NEW_LENGTH,
                                   temperature=1e-10,
                                   num_return_sequences=1)[0]['generated_text']
    generated_text = generated_text.replace(prompt, "")
    stop = ['--', '\n', ';', '#']
    stop_index = len(generated_text)
    for i, c in enumerate(generated_text):
        if c in stop:
            stop_index = i
            break
    return generated_text[:stop_index]
