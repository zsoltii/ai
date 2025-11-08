import os
import sys

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
# ---

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.log_available_gpus import log_available_gpus

log_available_gpus()
# Megjegyzés: a select_device() valószínűleg a fő eszközt választja ki.
# A device_map="auto" fogja elvégezni a modell elosztását.

model_id = "openai/gpt-oss-20b"

# --- A max_memory konfiguráció ellenőrzése ---
# Az Ön kódja már helyesen használja a max_memory-t, ami elosztja a terhelést a GPU-k
# (index 1 és 0) és a CPU között. A kulcsok a GPU indexeknek felelnek meg.
# 0: Radeon RX Vega (8GB) - max 7GiB
# 1: AMD Radeon RX 6900 XT (16GB) - max 14GiB
# max_memory = {
#     1: "14GiB", # GPU 1 (16GB VRAM)
#     0: "7GiB",  # GPU 0 (8GB VRAM)
#     "cpu": "30Gib" # Átkerülés a CPU RAM-ba, ha szükséges
# }

max_memory = {
    0: "10Gib", # Reduce memory allocation for GPU 0
    "cpu": "55Gib"
}

# Ensure an offload folder exists so Transformers can spill to disk if needed.
os.makedirs("./offload", exist_ok=True)

print("Using max_memory config:", max_memory)

# --- Modell Betöltése ---
# A device_map="auto" fogja automatikusan elosztani a rétegeket a max_memory és
# az offload_folder figyelembevételével, beleértve a CPU-t és a diszkre való
# ideiglenes mentést (offloading) is, ha szükséges.

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    max_memory=max_memory,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
    offload_folder="./offload", # Engedélyezi az ideiglenes mentést (offloading) diszkre, ha a CPU RAM elfogy.
)

# A kód további része változatlan...
# ...
print(f"VRAM elosztás: {max_memory}")

tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Modell betöltve. A modell elhelyezkedése:")
print(model.hf_device_map)

messages = [
    {"role": "system", "content": "Te egy segítőkész AI asszisztens vagy."},
    {"role": "user", "content": "Készíts egy rövid Python kódot, amely kiszámítja a Fibonacci sorozat első 10 elemét!"}
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Figyelem! A .to(model.device) itt a 'cuda' (fő device) lesz.
# Ha a modell részei a CPU-n vannak, de a bemenet a fő GPU-ra kerül,
# az adatok mozgatása ismét memóriaproblémát okozhat a generálás előtt.
# Bár a device_map="auto" kezeli a rétegeket, a bemenetnek a megfelelő
# eszközre kell kerülnie. Azonban a Hugging Face 'generate' metódusa
# gyakran átveszi a bemenetet a megfelelő eszközre. Maradjunk a model.device-nál.
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

print("Generálás folyamatban...")

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=256, do_sample=True, pad_token_id=tokenizer.eos_token_id)

print("decode folyamatban...")

response = tokenizer.decode(output[0].tolist())

print("AI válasz:" + response)