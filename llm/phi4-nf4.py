import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig

from util.log_available_gpus import log_available_gpus

log_available_gpus()

model_id = "microsoft/Phi-4-reasoning-plus"
# The multimodal model below has custom code that is incompatible with 4-bit quantization.
# model_id = "microsoft/Phi-4-multimodal-instruct"

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
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.half,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.uint8,
    llm_int8_enable_fp32_cpu_offload=True,  # Engedélyezi a CPU-ra történő offload-ot
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    quantization_config=quantization_config,
    device_map="auto",  # Engedélyezi az automatikus elosztást a GPU-k és CPU között.
    max_memory=max_memory,
    dtype=torch.half,
    low_cpu_mem_usage=False,
    offload_folder="./offload", # Engedélyezi az ideiglenes mentést (offloading) diszkre, ha a CPU RAM elfogy.
)

print(f"VRAM elosztás: {max_memory}")

tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Modell betöltve. A modell elhelyezkedése:")
print(model.hf_device_map)

messages = [
    {"role": "system", "content": "Te egy segítőkész AI asszisztens vagy. Mindig magyarul válaszolj!"},
    {"role": "user", "content": "Készíts egy rövid Python kódot, amely kiszámítja a Fibonacci sorozat első 10 elemét!"}
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

print("Generálás folyamatban...")

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=8096, do_sample=True, pad_token_id=tokenizer.eos_token_id)

print("decode folyamatban...")

response = tokenizer.decode(output[0].tolist())

print("AI válasz:" + response)