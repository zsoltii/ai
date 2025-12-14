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
MODEL_ID = "microsoft/Phi-4-mini-reasoning"
# A lokális adathalmazt tartalmazó főkönyvtár
dataset_path = "../wikiextractor/hu/huwiki_extracted/"
# A finomhangolt modell mentési neve (LoRA adapter)
new_model_name = "phi4-huwiki-finetuned"

# --- Kvantálási Konfiguráció (memóriahatékonyságért) ---
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# --- Modell betöltése ---
print(f"'{MODEL_ID}' modell betöltése 4-bites kvantálással...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# --- Tokenizer betöltése ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
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

model = get_peft_model(model, peft_config)
print("A modell felkészítve a PEFT (LoRA) tanításra.")

# --- Dokumentumok számának meghatározása ---
def count_documents(directory):
    """
    Gyorsan, memóriaterhelés nélkül megszámolja az érvényes dokumentumok számát.
    Minden fájlt egyetlen JSON objektumként kezel.
    """
    count = 0
    for root, _, files in os.walk(directory):
        files.sort()
        print(f"Processing root: {root}")
        print(f"Processing files: {files}")
        print(f"Processing files count: {files.__len__()}")
        for filename in files:
            if filename.startswith("wiki_"):
                filepath = os.path.join(root, filename)
                print(f"Counting file: {filepath}")
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        file_content = f.read() # Fájl tartalmának egyben beolvasása
                        data = json.loads(file_content) # JSON objektumként értelmezés
                        if 'text' in data and data['text']:
                            count += 1
                            print(f"Added for counting file: {filepath}")
                        else:
                            print(f"No text: {filepath}")
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON file {filepath} (JSONDecodeError)")
                except (KeyError, TypeError):
                    print(f"Warning: Skipping file {filepath} (missing 'text' field or TypeError)")
                except Exception as e:
                    print(f"Error counting file {filepath}: {e}")
            else:
                print(f"Skipping non-wiki file: {filename}")
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
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON file {filepath} (JSONDecodeError)")
                except (KeyError, TypeError):
                    print(f"Warning: Skipping file {filepath} (missing 'text' field or TypeError)")
                except Exception as e:
                    print(f"Error streaming file {filepath}: {e}")

dataset = IterableDataset.from_generator(stream_data_from_local_files, gen_kwargs={"directory": dataset_path})
print("Streaming adathalmaz beállítva.")

# --- Tanítási Argumentumok (Dinamikus lépésszámmal) ---
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
# num_epochs = 10 # ez az ideális, kb 90%-os pontosság érhető el vele, viszont 13-14 nap a magyar wikipediát feldolgozni egy AMD 6900 XT-vel
num_epochs = 2
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
trainer.train()
print("A finomhangolás befejeződött.")

# --- Modell mentése ---
print(f"A finomhangolt modell (adapter) mentése a '{new_model_name}' könyvtárba...")
trainer.model.save_pretrained(new_model_name)
print("Modell mentve.")
