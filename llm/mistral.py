import os
import sys
import time
from threading import Thread

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from util.log_available_gpus import log_available_gpus

log_available_gpus()

#model_id = "mistralai/Mistral-7B-Instruct-v0.2"
model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# max_memory = {
#     0: "14Gib", # Reduce memory allocation for GPU 0
#     "cpu": "55Gib"
# }

os.makedirs("./offload", exist_ok=True)

# print("Using max_memory config:", max_memory)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  # Engedélyezi az automatikus elosztást a GPU-k és CPU között.
    # max_memory=max_memory,
    dtype=torch.half,
    # low_cpu_mem_usage=False,
    # offload_folder="./offload", # Engedélyezi az ideiglenes mentést (offloading) diszkre, ha a CPU RAM elfogy.
)

# print(f"VRAM elosztás: {max_memory}")

tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Modell betöltve. A modell elhelyezkedése:")
print(model.hf_device_map)

messages = [
    {"role": "system", "content": "Te egy segítőkész AI asszisztens vagy."},
    {"role": "user", "content": "Készíts egy rövid Python kódot, amely kiszámítja a Fibonacci sorozat első 10 elemét!"}
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

print("Generálás folyamatban...")

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Generálási paraméterek
temperature = 1.2
top_p = 0.95
top_k = 60
repetition_penalty = 1.15
max_new_tokens = 4096

generation_kwargs = dict(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    streamer=streamer,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    repetition_penalty=repetition_penalty,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

start_time = time.time()
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

print("AI válasz:", end=" ", flush=True)

generated_text = ""
for new_text in streamer:
    print(new_text, end="", flush=True)
    generated_text += new_text

thread.join()
end_time = time.time()

print() # Új sor a végén

# Statisztika
generated_tokens = tokenizer.encode(generated_text)
num_tokens = len(generated_tokens)
duration = end_time - start_time
tps = num_tokens / duration if duration > 0 else 0

print(f"\n--- Statisztika ---")
print(f"Generált tokenek száma: {num_tokens}")
print(f"Sebesség: {tps:.2f} token/másodperc")
print(f"Időtartam: {duration:.2f} másodperc")
print(f"Temperature: {temperature}")
print(f"Top P: {top_p}")
print(f"Top K: {top_k}")
print(f"Repetition Penalty: {repetition_penalty}")
print(f"EOS Token ID: {tokenizer.eos_token_id}")
