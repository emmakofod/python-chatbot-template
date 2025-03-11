from transformers import pipeline
import torch

chatbot = pipeline(
    "text-generation",
    model="mistralai/Mistral-Small-24B-Instruct-2501",
    max_new_tokens=256,
    torch_dtype=torch.bfloat16
)

messages = [
    {"role": "user", "content": "Give me 5 non-formal ways to say 'See you later' in French."}
]

response = chatbot(messages)
print(response)
