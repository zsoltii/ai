import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.log_available_gpus import log_available_gpus
from util.select_device import select_device

log_available_gpus()
device = select_device()

model_id = "openai/gpt-oss-20b"
#
# max_memory = {
#     1: "14GiB", # Reduce memory allocation for GPU 1
#     0: "7GiB",  # Reduce memory allocation for GPU 0
#     "cpu": "30Gib"
# }

max_memory = {
    0: "14GiB", # Reduce memory allocation for GPU 0
    "cpu": "30Gib"
}

# Ensure an offload folder exists so Transformers can spill to disk if needed.
os.makedirs("./offload", exist_ok=True)

print("Using max_memory config:", max_memory)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  # Enable automatic device mapping
    max_memory=max_memory,
    dtype=torch.bfloat16,  # Use bfloat16 for better ROCm compatibility
    low_cpu_mem_usage=True,
    offload_folder="./offload",
)

# Log the device being used
print("Model loaded on device:", device)
print(f"VRAM elosztás: {max_memory}")

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

output = model.generate(**inputs, max_new_tokens=1024, do_sample=True)

print("decode folyamatban...")

response = tokenizer.decode(output[0].tolist())

print("AI válasz:" + response)