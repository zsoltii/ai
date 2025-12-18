import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math, json, torch
from datasets import IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig

from util.log_available_gpus import log_available_gpus
from util.hfh_login import hfh_login
from util.finetune import should_resume_from_checkpoint, create_new_model_name, get_last_checkpoint, QUANTIZATION_CONFIG, RESULTS_DIRECTORY, PEFT_CONFIG

hfh_login()
log_available_gpus()

# --- Modell és Tokenizer beállítások ---
BASE_MODEL_ID = "microsoft/Phi-4-mini-reasoning"
# A finomhangolt modell mentési neve (LoRA adapter)
NEW_MODEL_NAME = create_new_model_name(BASE_MODEL_ID, "finetuned")
# A finomhangolt LoRA adapter könyvtára
ADAPTER_MODEL_PATH = "./" + NEW_MODEL_NAME
# A lokális adathalmazt tartalmazó főkönyvtár
DATASET_PATH = "../wikiextractor/hu/huwiki_extracted/"
# DATASET_PATH = "../wikiextractor/hu/teszt"

PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
# NUM_EPOCHS = 10 # ez az ideális, kb 90%-os pontosság érhető el vele, viszont 13-14 nap a magyar wikipediát feldolgozni egy AMD 6900 XT-vel
NUM_EPOCHS = 4

# --- Memória és Offload beállítások ---
MAX_MEMORY = {
    0: "14Gib", # Memória korlát a 0-s GPU-ra
    "cpu": "85Gib" # Memória korlát a CPU-ra (offload esetén)
}
os.makedirs("./offload", exist_ok=True)
print("max_memory konfiguráció használata:", MAX_MEMORY)



# --- Modell betöltése ---
print(f"'{BASE_MODEL_ID}' modell betöltése 4-bites kvantálással...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=QUANTIZATION_CONFIG,
    device_map="auto",
    max_memory=MAX_MEMORY,
    dtype=torch.half,
    low_cpu_mem_usage=False,
    offload_folder="./offload",
    trust_remote_code=True,
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# --- Alapértelmezett Tokenizer betöltése ---
print("Alapértelmezett Tokenizer betöltése")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

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


# --- Teljes adathalmaz finomhangolása ---
output_dir = RESULTS_DIRECTORY + "-" + NEW_MODEL_NAME
print(f"--- Teljes adathalmaz feldolgozása: {DATASET_PATH} ---")
print(f"--- Eredmények mentése ide: {output_dir} ---")

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

print("\nModell betöltve. A modell elhelyezkedése:")
print(model.hf_device_map)

print("Dokumentumok számának meghatározása...")
num_documents = count_documents(DATASET_PATH)
if num_documents == 0:
    raise ValueError("Nem található feldolgozható adat a megadott könyvtárban.")
print(f"Talált dokumentumok száma: {num_documents}")

dataset = IterableDataset.from_generator(stream_data_from_local_files, gen_kwargs={"directory": DATASET_PATH})
print("Streaming adathalmaz beállítva.")

# --- Tanítási Argumentumok (Dinamikus lépésszámmal) ---
effective_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
steps_per_epoch = math.ceil(num_documents / effective_batch_size)
max_steps = steps_per_epoch * NUM_EPOCHS

# A mentési gyakoriság beállítása a kérésnek megfelelően
save_steps = max(1, min(steps_per_epoch // 2, 6))

print(f"Dinamikusan számított max_steps: {max_steps} ({NUM_EPOCHS} epoch-hoz)")
print(f"Mentési gyakoriság (save_steps): {save_steps}")

training_arguments = SFTConfig(
    output_dir=output_dir,
    max_steps=max_steps,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
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
    save_total_limit=2,
)

# --- Tréner inicializálása ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=PEFT_CONFIG,
    processing_class=tokenizer,
    args=training_arguments,
)

# --- Tanítás indítása ---
print("A finomhangolás elindítása...")
# A resume_from_checkpoint=True argumentum biztosítja, hogy a Trainer
# automatikusan betöltse a legutóbbi checkpointot, ha létezik.
trainer.train(resume_from_checkpoint=should_resume_from_checkpoint(output_dir))
print("A finomhangolás befejeződött.")

# --- Modell mentése ---
print(f"A finomhangolt modell (adapter) mentése a '{NEW_MODEL_NAME}' könyvtárba...")
trainer.model.save_pretrained(NEW_MODEL_NAME)
# trainer.save_model(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)
print("Modell mentve.")

# --- Memória felszabadítása az összefésülés előtt ---
print("Memória felszabadítása az összefésülés előtt...")
del model, base_model
del trainer
torch.cuda.empty_cache()
print("Memória felszabadítva.")


# --- Önállóan betölthető modell mentése ---
print("A LoRA adapter és a bázismodell összefésülése...")

# A bázismodell újratöltése kvantálással, hogy elférjen a memóriában
merged_model_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=QUANTIZATION_CONFIG,
    device_map="auto",
    max_memory=MAX_MEMORY,
    dtype=torch.half,
    low_cpu_mem_usage=False,
    offload_folder="./offload",
    trust_remote_code=True,
)

# A finomhangolt LoRA adapter betöltése
# A `PeftModel` automatikusan kezeli a kvantált bázismodellt
peft_model = PeftModel.from_pretrained(merged_model_base, NEW_MODEL_NAME)

# Az adapter súlyainak összefésülése a bázismodellel
# A `merge_and_unload` metódus a kvantált modellen is működik
merged_model = peft_model.merge_and_unload()
print("Az összefésülés befejeződött.")

# Az összefésült modell mentése
MERGED_MODEL_PATH = f"{NEW_MODEL_NAME}-merged"
print(f"Az összefésült, önállóan betölthető modell mentése a '{MERGED_MODEL_PATH}' könyvtárba...")
merged_model.save_pretrained(MERGED_MODEL_PATH)
tokenizer.save_pretrained(MERGED_MODEL_PATH)

print("Az önálló modell mentése befejeződött.")
