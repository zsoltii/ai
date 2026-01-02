import os, sys

# Figyelmeztetés: A gfx908 nem a VEGA64 architektúrája.
# A VEGA64 a 'gfx900' (9.0.0) architektúrát használja.
# A 'gfx908' (9.0.8) egy másik GPU-hoz (AMD Instinct MI100) tartozik.
# Ennek a beállításnak az erőltetése hibát okozhat.
# os.environ['HSA_OVERRIDE_GFX_VERSION'] = '9.0.a'

# Ezt a részt megtartjuk a környezet beállításához
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

from util.log_available_gpus import log_available_gpus
from util.hfh_login import hfh_login

hfh_login()
log_available_gpus()

model_id = "meta-llama/Llama-3.1-8B"

max_memory = {
    # 0: "8GIB",
    0: "13Gib", # Reduce memory allocation for GPU 0
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

print("Modell betöltve. A modell elhelyezkedése:")
print(model.hf_device_map)

# Pipeline inicializálása
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # Fontos, hogy a 'return_full_text' False legyen, ha csak a generált választ akarjuk
    #return_full_text=False,
    # További beállítások:
    # do_sample=True, # Engedélyezi a mintavételezést
    # top_p=0.9,
    # temperature=0.6,
)

USER_PROMPT = "Készíts egy rövid Python kódot, amely kiszámítja a Fibonacci sorozat első 10 elemét!"

print("-" * 50)
print("📥 Generált Bemeneti Prompt (Llama formátum):")
print(USER_PROMPT)
print("-" * 50)

print("⏳ Válasz generálása...")

result = pipe(USER_PROMPT, max_new_tokens=4096, do_sample=True, pad_token_id=tokenizer.eos_token_id)

print("✅ Generált Válasz:")
print(result)
print("-" * 50)
