import os
import sys
import time
import shutil
import torch
from datasets import load_dataset, concatenate_datasets, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.log_available_gpus import log_available_gpus
from util.select_device import select_device
from util.hfh_login import hfh_login

# Set environment variable to reduce memory fragmentation
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

log_available_gpus()
device = select_device()
hfh_login()

# --- Konfiguráció ---
MODEL = "Helsinki-NLP/opus-mt-tc-big-en-hu"
INPUT_FILE = "../text/hu/csv/en-pl.csv"
OUTPUT_FILE = "../text/hu/csv/en-hu-opus-mt-tc-big-2.csv"
BATCH_SIZE = 224  # VRAM-tól függően állítható
MAX_LENGTH = 512 # A modell maximális input hossza
SHARD_SIZE = BATCH_SIZE * 10 # Ennyi soronként mentünk állapotot (checkpoint)
TEMP_DIR = "temp_translation_shards" # Ideiglenes mappa a részeredményeknek

# --- Modell és Tokenizer betöltése ---
print("Tokenizer és modell betöltése (eltarthat egy ideig)...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
).to(device)

# --- SPEEDUP: torch.compile használata a jelentős gyorsulásért ---
print("Modell fordítása torch.compile()-lal... (az első futtatás lassabb lesz)")
model = torch.compile(model, mode="reduce-overhead")

print(f"Modell betöltve: {getattr(model, 'name_or_path', str(model))}")
print("Tokenizer előkészítése...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
print("Tokenizer előkészítve.")

if hasattr(model, "hf_device_map"):
    print("Modell eszköz kiosztása:")
    print(model.hf_device_map)

print("-------------------------------------------------------------------------------------------------")


# --- Fordítási függvény a dataset.map számára ---
def translate_batch(batch):
    """
    Lefordítja a bemeneti 'en' oszlopot és hozzáadja az eredményt 'hu' oszlopként.
    """
    try:
        en_sentences = batch["en"]
        inputs = tokenizer(
            en_sentences,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_LENGTH,
                num_beams=1,
                do_sample=False,
                use_cache=True,
            )

        hu_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        batch["hu"] = hu_sentences
    except Exception as e:
        print(f"Hiba a kötegelt fordítás során: {e}")
        # Hiba esetén üres stringekkel töltjük fel, hogy a struktúra megmaradjon
        batch["hu"] = [""] * len(batch["en"])
    return batch


# --- Fő logika ---
print("Folyamat indítása...")
translate_start = time.time()

# Adatállomány betöltése a datasets könyvtárral
print(f"Adatállomány betöltése: {INPUT_FILE}...")
dataset = load_dataset("csv", data_files=INPUT_FILE, split="train")
total_rows = len(dataset)
print(f"Összes sor: {total_rows}")

# Kiszámoljuk, hány darabra kell bontani az adatbázist
num_shards = (total_rows + SHARD_SIZE - 1) // SHARD_SIZE
print(f"Az adatbázis {num_shards} részre (shard) lesz osztva a biztonságos mentés érdekében.")

# Ideiglenes mappa létrehozása
os.makedirs(TEMP_DIR, exist_ok=True)

processed_shards = []

for i in range(num_shards):
    shard_path = os.path.join(TEMP_DIR, f"shard_{i}")
    
    # Ellenőrizzük, hogy ez a rész már kész van-e
    if os.path.exists(shard_path):
        try:
            print(f"[{i+1}/{num_shards}] Részlet betöltése a lemezről (már elkészült)...")
            ds_shard = load_from_disk(shard_path)
            processed_shards.append(ds_shard)
            continue
        except Exception as e:
            print(f"[{i+1}/{num_shards}] Hiba a mentett részlet betöltésekor, újrafeldolgozás: {e}")
            shutil.rmtree(shard_path, ignore_errors=True)
    
    # Ha nincs kész, feldolgozzuk
    print(f"[{i+1}/{num_shards}] Részlet feldolgozása...")
    
    # Kiválasztjuk az aktuális szeletet (contiguous=True fontos a sorrend miatt)
    shard_input = dataset.shard(num_shards=num_shards, index=i, contiguous=True)
    
    # Fordítás
    shard_processed = shard_input.map(
        translate_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        desc=f"Fordítás ({i+1}/{num_shards})"
    )
    
    # Mentés lemezre
    print(f"[{i+1}/{num_shards}] Részlet mentése ide: {shard_path}...")
    shard_processed.save_to_disk(shard_path)
    processed_shards.append(shard_processed)

# Összefűzés
print("Az összes részlet elkészült. Összefűzés...")
translated_dataset = concatenate_datasets(processed_shards)

# Felesleges oszlop eltávolítása és mentés CSV-be
print("Eredmény mentése CSV-be...")
if "pl" in translated_dataset.column_names:
    translated_dataset = translated_dataset.remove_columns("pl")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
translated_dataset.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

# Takarítás: ha sikeres volt a mentés, töröljük az ideiglenes fájlokat
print("Ideiglenes fájlok törlése...")
try:
    shutil.rmtree(TEMP_DIR)
except Exception as e:
    print(f"Figyelem: Nem sikerült törölni az ideiglenes mappát ({TEMP_DIR}): {e}")

translate_end = time.time()
translate_duration = translate_end - translate_start

print(f"\nFordítás befejezve. Eredmény mentve ide: {OUTPUT_FILE}")
print(f"A futás teljes ideje: {translate_duration:.2f} másodperc.")
print("Kész.")
