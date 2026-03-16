import os
import sys
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig, TextIteratorStreamer
from peft import LoraConfig, get_peft_model, PeftModel

# --- Környezet beállítása ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.log_available_gpus import log_available_gpus
from util.hfh_login import hfh_login
from util.finetune import create_new_model_name, get_last_checkpoint, QUANTIZATION_CONFIG, RESULTS_DIRECTORY, PEFT_CONFIG

hfh_login()
log_available_gpus()

# --- Konfiguráció ---
# Az alapmodell, amire a finomhangolást végeztük
BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
# A finomhangolt modell mentési neve (LoRA adapter)
NEW_MODEL_NAME = create_new_model_name(BASE_MODEL_ID, "finetuned")
# A finomhangolt LoRA adapter könyvtára
ADAPTER_MODEL_PATH = "./" + NEW_MODEL_NAME

# --- Memória és Offload beállítások ---
# max_memory = {
#     0: "14Gib", # Memória korlát a 0-s GPU-ra
#     "cpu": "85Gib" # Memória korlát a CPU-ra (offload esetén)
# }
os.makedirs("./offload", exist_ok=True)
# print("Using max_memory config:", max_memory)

# --- Kvantálási Konfiguráció (részletes) ---
# A modell eredeti kvantálási sémájának felülbírálása a konzisztencia érdekében
config = AutoConfig.from_pretrained(BASE_MODEL_ID)
if hasattr(config, "quantization_config"):
    print(f"Original quantization_config {config.quantization_config}")
    del config.quantization_config
    print("Original quantization_config deleted!")

# --- 1. Lépés: Alapmodell betöltése (részletes paraméterekkel) ---
print(f"Alapmodell betöltése: '{BASE_MODEL_ID}'")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    config=config,
    quantization_config=QUANTIZATION_CONFIG,
    device_map="auto",
    # max_memory=max_memory,
    dtype=torch.half,
    low_cpu_mem_usage=False,
    offload_folder="./offload",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# --- 2. Lépés: Adapter betöltése és a modellek egyesítése ---
output_dir = RESULTS_DIRECTORY + "-" + NEW_MODEL_NAME

model = base_model

adapter_path = os.path.join(ADAPTER_MODEL_PATH, "adapter_model.safetensors")
is_adapter_saved = os.path.exists(adapter_path)
last_checkpoint = get_last_checkpoint(output_dir)

if last_checkpoint:
    checkpoint_path = os.path.join(output_dir, last_checkpoint)
    print(f"Meglévő adapter és toknaizer betöltése a '{checkpoint_path}' könyvtárból a tanítás folytatásához...")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    print("Adapter sikeresen betöltve.")
elif is_adapter_saved:
    print(f"Meglévő adapter és tokanizer betöltése a '{ADAPTER_MODEL_PATH}' könyvtárból a tanítás folytatásához...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL_PATH, trust_remote_code=True)
    print("Adapter sikeresen betöltve.")
else:
    print("Nem található meglévő adapter. Új adapter létrehozása...")
    model = get_peft_model(base_model, PEFT_CONFIG)
    print("Új adapter sikeresen létrehozva.")
    print("A modell felkészítve a PEFT (LoRA) tanításra.")

# print("\nModell betöltve. A modell elhelyezkedése:")
# print(model.hf_device_map)

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
