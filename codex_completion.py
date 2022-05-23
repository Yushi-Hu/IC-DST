import openai
from config import CONFIG

"""Codex completion"""

def codex_completion(prompt_text):
    openai.api_key = CONFIG['openai_api_key']
    return openai.Completion.create(
        engine='code-davinci-002',
        prompt=prompt_text,
        max_tokens=200,
        temperature=0,
        stop=['--', '\n', ';', '#'],
    )["choices"][0]["text"]
