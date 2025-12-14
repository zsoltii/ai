import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import json
import torch
from datasets import IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from util.log_available_gpus import log_available_gpus
from util.hfh_login import hfh_login

hfh_login()
log_available_gpus()

# --- Modell és Tokenizer beállítások ---
# MODEL_ID = "microsoft/Phi-4-reasoning-plus"
BASE_MODEL_ID = "microsoft/Phi-4-mini-reasoning"
# A finomhangolt modell mentési neve (LoRA adapter)
NEW_MODEL_NAME = BASE_MODEL_ID.replace("/", "-") + "-finetuned"
# A finomhangolt LoRA adapter könyvtára
ADAPTER_MODEL_PATH = "./" + NEW_MODEL_NAME
# A lokális adathalmazt tartalmazó főkönyvtár
dataset_path = "../wikiextractor/hu/huwiki_extracted/AA"

# --- Memória és Offload beállítások ---
max_memory = {
    0: "14Gib", # Memória korlát a 0-s GPU-ra
    "cpu": "85Gib" # Memória korlát a CPU-ra (offload esetén)
}
os.makedirs("./offload", exist_ok=True)
print("max_memory konfiguráció használata:", max_memory)

# --- Kvantálási Konfiguráció (memóriahatékonyságért) ---
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# --- Modell betöltése ---
print(f"'{BASE_MODEL_ID}' modell betöltése 4-bites kvantálással...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    max_memory=max_memory,
    dtype=torch.half,
    low_cpu_mem_usage=False,
    offload_folder="./offload",
    trust_remote_code=True,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# --- Tokenizer betöltése ---
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- PEFT (LoRA) Konfiguráció ---
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj"
    ]
)

# A modellt mindig előkészítjük a PEFT-re. A Trainer fogja eldönteni, hogy betölt-e egy checkpointot.
model = get_peft_model(model, peft_config)
print("A modell felkészítve a PEFT (LoRA) tanításra.")


# --- Dokumentumok számának meghatározása ---
def count_documents(directory):
    """
    Gyorsan, memóriaterhelés nélkül megszámolja az érvényes dokumentumok számát.
    Minden fájlt egyetlen JSON objektumként kezel.
    """
    count = 0
    skip_count = 0
    for root, _, files in os.walk(directory):
        files.sort()
        print(f"Feldolgozandó könyvtár: {root}")
        # print(f"Feldolgozandó fájlok: {files}")
        print(f"Fájlok száma: {files.__len__()}")
        for filename in files:
            if filename.startswith("wiki_"):
                filepath = os.path.join(root, filename)
                # print(f"Fájl számolása: {filepath}")
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        file_content = f.read() # Fájl tartalmának egyben beolvasása
                        data = json.loads(file_content) # JSON objektumként értelmezés
                        if 'text' in data and data['text']:
                            count += 1
                            # print(f"Fájl hozzáadva a számoláshoz: {filepath}")
                        else:
                            # print(f"Nincs szöveg: {filepath}")
                            skip_count += 1
                except json.JSONDecodeError:
                    print(f"Figyelmeztetés: Érvénytelen JSON fájl kihagyva {filepath} (JSONDecodeError)")
                    skip_count += 1
                except (KeyError, TypeError):
                    print(f"Figyelmeztetés: Fájl kihagyva {filepath} (hiányzó 'text' mező vagy TypeError)")
                    skip_count += 1
                except Exception as e:
                    print(f"Hiba a fájl számolása közben {filepath}: {e}")
                    skip_count += 1
            else:
                print(f"Nem-wiki fájl kihagyva: {filename}")
    print(f"Kihagyott fájlok száma: {skip_count}")
    return count

print("Dokumentumok számának meghatározása...")
num_documents = count_documents(dataset_path)
if num_documents == 0:
    raise ValueError("Nem található feldolgozható adat a megadott könyvtárban.")
print(f"Talált dokumentumok száma: {num_documents}")

# --- Adatfolyam (Streaming) beállítása ---
def stream_data_from_local_files(directory):
    """
    Generátor függvény, amely bejárja a könyvtárakat és fájlonként,
    egy JSON objektumként "yieldeli" a feldolgozott adatokat.
    """
    for root, _, files in os.walk(directory):
        files.sort()
        for filename in files:
            if filename.startswith("wiki_"):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        file_content = f.read() # Fájl tartalmának egyben beolvasása
                        data = json.loads(file_content) # JSON objektumként értelmezés
                        if 'text' in data and data['text']:
                            structured_text = (
                                f"Title: {data.get('title', '')}\n"
                                f"ID: {data.get('id', '')}\n"
                                f"Revision ID: {data.get('revid', '')}\n"
                                f"URL: {data.get('url', '')}\n\n"
                                f"{data['text']}"
                            )
                            yield {"text": structured_text}
                except Exception as e:
                    print(f"Hiba a fájl streamelése közben {filepath}: {e}")

dataset = IterableDataset.from_generator(stream_data_from_local_files, gen_kwargs={"directory": dataset_path})
print("Streaming adathalmaz beállítva.")

# --- Tanítási Argumentumok (Dinamikus lépésszámmal) ---
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
# num_epochs = 10 # ez az ideális, kb 90%-os pontosság érhető el vele, viszont 13-14 nap a magyar wikipediát feldolgozni egy AMD 6900 XT-vel
num_epochs = 6
effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
steps_per_epoch = math.ceil(num_documents / effective_batch_size)
max_steps = steps_per_epoch * num_epochs

# A mentési gyakoriság beállítása a kérésnek megfelelően
save_steps = min(steps_per_epoch // 2, 100)

print(f"Dinamikusan számított max_steps: {max_steps} ({num_epochs} epoch-hoz)")
print(f"Mentési gyakoriság (save_steps): {save_steps}")


training_arguments = SFTConfig(
    output_dir="./results",
    max_steps=max_steps,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    save_steps=save_steps,
    logging_steps=save_steps,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    dataset_text_field="text",
    max_length=1024,
    seed=42,
)

# --- Tréner inicializálása ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_arguments,
)

# --- Tanítás indítása ---
print("A finomhangolás elindítása...")
# A resume_from_checkpoint=True argumentum biztosítja, hogy a Trainer
# automatikusan betöltse a legutóbbi checkpointot, ha létezik.
trainer.train(resume_from_checkpoint=True)
print("A finomhangolás befejeződött.")

# --- Modell mentése ---
print(f"A finomhangolt modell (adapter) mentése a '{NEW_MODEL_NAME}' könyvtárba...")
trainer.model.save_pretrained(NEW_MODEL_NAME)
print("Modell mentve.")
