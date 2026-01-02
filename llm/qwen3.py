import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig, TextIteratorStreamer

from util.finetune import QUANTIZATION_CONFIG
from util.log_available_gpus import log_available_gpus

# log_available_gpus()

model_id = "Qwen/Qwen3-1.7B" # TPS: 6-7s/it; ~1100h;
# model_id = "Qwen/Qwen3-4B" # TPS: 19-20s/it; ~3000h
# model_id = "Qwen/Qwen3-8B" # TPS: 28-29s/it; ~4600h

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
    print("Original quantization_config deleted!")

# Létrehozunk egy új, 4-bites kvantálási konfigurációt, amely támogatja a CPU offload-ot.

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # config=config,
    # quantization_config=QUANTIZATION_CONFIG,
    device_map="auto",  # Engedélyezi az automatikus elosztást a GPU-k és CPU között.
    max_memory=max_memory,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
    offload_folder="./offload", # Engedélyezi az ideiglenes mentést (offloading) diszkre, ha a CPU RAM elfogy.
)

print(f"VRAM elosztás: {max_memory}")

tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Modell betöltve. A modell elhelyezkedése:")
print(model.hf_device_map)

messages = [
    {"role": "system", "content": "Te egy segítőkész AI asszisztens vagy. Mindig magyarul válaszolj!"},
    {"role": "user", "content": "Írj egy kedves gyerekeknek szóló történetet!"}
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

print("\nGenerálás folyamatban a finomhangolt modellel...")

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=1024*4, do_sample=True)

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

print("\n--- AI VÁLASZ ---")
for new_text in streamer:
    print(new_text, end="", flush=True)
print()