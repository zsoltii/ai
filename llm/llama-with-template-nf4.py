import os, sys

# Ezt a részt megtartjuk a környezet beállításához
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM,  BitsAndBytesConfig, AutoConfig

# Feltételezve, hogy ez a segédprogram bejelentkezik a Hugging Face-re
from util.hfh_login import hfh_login

hfh_login()

# model_id = "meta-llama/Llama-3.1-8B"
model_id = "meta-llama/Llama-3.1-8B-Instruct"
#model_id = "meta-llama/Llama-3.1-70B"
#model_id = "meta-llama/Llama-3.1-70B-Instruct"

# model_id = "meta-llama/Llama-4-Scout-17B-16E" # too big

max_memory = {
    0: "14Gib", # Reduce memory allocation for GPU 0
    "cpu": "55Gib"
}

os.makedirs("./offload", exist_ok=True)

print("Using max_memory config:", max_memory)

# --- Kvantálási Konfiguráció ---
# A modell eredeti kvantálási sémájának felülbírálása.
config = AutoConfig.from_pretrained(model_id)

if hasattr(config, "quantization_config"):
    print(f"Original quantization_config {config.quantization_config}")
    del config.quantization_config
    print(f"Original quantization_config deleted!")

# Létrehozunk egy új, 4-bites kvantálási konfigurációt, amely támogatja a CPU offload-ot.
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.half,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.uint8,
    llm_int8_enable_fp32_cpu_offload=True,  # Engedélyezi a CPU-ra történő offload-ot
)

print("Loading tokenizer and model (this may take a while)...")

# Modell betöltése
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    quantization_config=quantization_config,
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

# Pipeline inicializálása
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # Fontos, hogy a 'return_full_text' False legyen, ha csak a generált választ akarjuk
    return_full_text=False,
    # További beállítások:
    # do_sample=True, # Engedélyezi a mintavételezést
    # top_p=0.9,
    # temperature=0.6,
)


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

result = pipe(prompt, max_new_tokens=4096, do_sample=True, pad_token_id=tokenizer.eos_token_id)

generated_text = result[0]['generated_text']

print("✅ Generált Válasz:")
print(generated_text)
print("-" * 50)
