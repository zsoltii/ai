import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Környezet beállítása ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.log_available_gpus import log_available_gpus
from util.hfh_login import hfh_login
from util.finetune import create_new_model_name

hfh_login()
log_available_gpus()

# --- Konfiguráció ---
# Az alapmodell, amire a finomhangolást végeztük
BASE_MODEL_ID = "microsoft/Phi-4-mini-reasoning"
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
    dtype=torch.half,
    low_cpu_mem_usage=False,
    offload_folder="./offload",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

print("\nModell betöltve. A modell elhelyezkedése:")
print(model.hf_device_map)

# --- 3. Lépés: Inferencia a finomhangolt modellel ---
# messages = [
#     {"role": "system", "content": "Te egy segítőkész AI asszisztens vagy. Mindig magyarul válaszolj!"},
#     {"role": "user", "content": "Ki volt a legelső főispánja Heves és Külső-Szolnok vármegyének és mettől meddig töltötte be ezt a tisztséget?"}
# ]

messages = [
    {"role": "system", "content": "You are a random sentence generator. Your output must consist of exactly one grammatically correct English sentence. Output ONLY the sentence itself. The sentence must be in English. No thinking! As fast as possible!"},
    {"role": "user", "content": "Generate one random English sentence. No thinking, just the sentence."}
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

print("\nGenerálás folyamatban a finomhangolt modellel...")

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=1024*32, do_sample=True, temperature=0.9, top_p=0.5, pad_token_id=tokenizer.eos_token_id)

print("Dekódolás folyamatban...")

response = tokenizer.decode(output[0], skip_special_tokens=True)

print("\n--- AI VÁLASZ ---")
print(response)
