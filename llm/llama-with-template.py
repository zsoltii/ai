import os, sys
import time
from threading import Thread

# Ezt a részt megtartjuk a környezet beállításához
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# Feltételezve, hogy ez a segédprogram bejelentkezik a Hugging Face-re
from util.hfh_login import hfh_login

hfh_login()

model_id = "meta-llama/Llama-3.1-8B"

max_memory = {
    0: "14Gib", # Reduce memory allocation for GPU 0
    "cpu": "55Gib"
}

os.makedirs("./offload", exist_ok=True)

print("Using max_memory config:", max_memory)

print("Loading tokenizer and model (this may take a while)...")

# Modell betöltése
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.half,
    device_map="auto",
    max_memory=max_memory,
    offload_folder="./offload",
    low_cpu_mem_usage=False,
)

print(f"VRAM elosztás: {max_memory}")

# Tokenizer betöltése
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)

# FIXME: kitenni külön library-ba
LLAMA_3_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{{ message['content'] }}<|eot_id|>"
    "{% elif message['role'] == 'user' %}"
    "<|start_header_id|>user<|end_header_id|>\n{{ message['content'] }}<|eot_id|>"
    "{% elif message['role'] == 'assistant' %}"
    "<|start_header_id|>assistant<|end_header_id|>\n{{ message['content'] }}<|eot_id|>"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n{% endif %}"
)

tokenizer.chat_template = LLAMA_3_TEMPLATE

print("Modell betöltve. A modell elhelyezkedése:")
print(model.hf_device_map)


SYSTEM_PROMPT = "Egy segítőkész és barátságos mesterséges intelligencia asszisztens vagy."
USER_PROMPT = "Készíts egy rövid Python kódot, amely kiszámítja a Fibonacci sorozat első 10 elemét!"

# Beszélgetés formázása a Llama formátum szerint (Chat Template)
# A 'messages' egy listát vár, ahol minden elem egy dictionary: {"role": "...", "content": "..."}
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_PROMPT}
]

# Az apply_chat_template formázza az üzeneteket a modell által elvárt formátumra
# (pl. [INST] <<SYS>> system prompt <<SYS>> user prompt [/INST])
# 'tokenize=False' biztosítja, hogy a kimenet egy string (szöveg) legyen
# 'add_generation_prompt=True' hozzáadja a hiányzó generációs promptot (pl. a legutolsó promptot)
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("-" * 50)
print("📥 Generált Bemeneti Prompt (Llama formátum):")
print(prompt)
print("-" * 50)

print("⏳ Válasz generálása...")

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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
