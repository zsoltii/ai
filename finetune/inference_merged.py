import os
import sys
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# --- Környezet beállítása ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.log_available_gpus import log_available_gpus
from util.hfh_login import hfh_login
from util.finetune import create_new_model_name

# hfh_login()
# log_available_gpus()

# --- Konfiguráció ---
# Az alapmodell, amire a finomhangolást végeztük
# BASE_MODEL_ID = "microsoft/Phi-4-mini-reasoning"
BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
# A finomhangolt modell mentési neve (LoRA adapter)
MERGED_MODEL_NAME = create_new_model_name(BASE_MODEL_ID, "finetuned-merged")
# A finomhangolt LoRA adapter könyvtára
MERGED_MODEL_PATH = "./" + MERGED_MODEL_NAME

# --- Memória és Offload beállítások ---
max_memory = {
    0: "14Gib", # Memória korlát a 0-s GPU-ra
    "cpu": "85Gib" # Memória korlát a CPU-ra (offload esetén)
}
os.makedirs("./offload", exist_ok=True)
print("Using max_memory config:", max_memory)

# --- Kvantálási Konfiguráció (részletes) ---

# --- 1. Lépés: Alapmodell betöltése (részletes paraméterekkel) ---
print(f"Modell betöltése: '{MERGED_MODEL_PATH}'")
model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_PATH,
    device_map="auto",
    max_memory=max_memory,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
    offload_folder="./offload",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
# tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

print("\nModell betöltve. A modell elhelyezkedése:")
print(model.hf_device_map)

messages = [
    {"role": "system", "content": "Te egy segítőkész AI asszisztens vagy. Mindig magyarul válaszolj!"},
    {"role": "user", "content": "Írj egy kedves gyerekeknek szóló nagyon rövid történetet!"}
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
