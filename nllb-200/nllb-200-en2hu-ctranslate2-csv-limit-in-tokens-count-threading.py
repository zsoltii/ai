import os
import sys
import csv
import time
import shutil
import torch
import threading
import queue
import concurrent.futures
from tqdm import tqdm
import ctranslate2
from transformers import AutoTokenizer
from ctranslate2.converters import TransformersConverter

# --- Configuration Constants ---

# Maximális szószám egy mondatban. Ennél hosszabb mondatokat a script üresre cserél és nem fordít le.
MAX_WORDS_LENGTH = 450

# A Hugging Face modell azonosítója (NLLB-200 distilled 1.3B változat).
MODEL_NAME = "facebook/nllb-200-distilled-1.3B"

# A konvertált CTranslate2 modell mentési helye (könyvtár).
CT2_MODEL_PATH = "nllb-200-distilled-1.3B-ct2"

# A forrásnyelv kódja (NLLB formátum).
SOURCE_LANGUAGE = "eng_Latn"

# A célnyelv kódja (NLLB formátum).
TARGET_LANGUAGE = "hun_Latn"

# A modell kvantálási típusa a konverzió során (pl. int8 a gyorsabb futásért és kisebb memóriaigényért).
QUANTIZATION = "int8"

# A bemeneti CSV fájl elérési útja.
INPUT_FILE = "../text/hu/csv/en-pl.csv"

# A kimeneti CSV fájl elérési útja.
OUTPUT_FILE = "../text/hu/csv/en-hu-nllb-200-ct2.csv"

# A fordítási batch maximális mérete tokenekben.
MAX_BATCH_TOKENS = 250

# Szorzó a chunk méret meghatározásához a többszálú tokenizálásnál (chunk_size = cpu_count * multiplier).
CHUNK_SIZE_MULTIPLIER = 10

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.log_available_gpus import log_available_gpus
from util.hfh_login import hfh_login

# Set environment variable to reduce memory fragmentation
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

hfh_login()

# Determine device for CTranslate2
try:
    cuda_count = ctranslate2.get_cuda_device_count()
except Exception:
    cuda_count = 0

device_type = "cuda" if cuda_count > 0 else "cpu"
print(f"CTranslate2 detected {cuda_count} CUDA devices. Initial selection: {device_type}")

# Check if conversion is needed
if not os.path.exists(CT2_MODEL_PATH):
    print(f"Converting model {MODEL_NAME} to CTranslate2 format...")
    converter = TransformersConverter(MODEL_NAME)
    converter.convert(output_dir=CT2_MODEL_PATH, quantization=QUANTIZATION, force=True)
    print(f"Model converted to {CT2_MODEL_PATH} with quantization {QUANTIZATION}.")
else:
    pass

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.src_lang = SOURCE_LANGUAGE

print(f"Loading CTranslate2 translator on {device_type}...")
try:
    if device_type == "cpu":
        translator = ctranslate2.Translator(
            CT2_MODEL_PATH,
            device=device_type,
            compute_type=QUANTIZATION,
            intra_threads=os.cpu_count()
        )
    else:
        translator = ctranslate2.Translator(
            CT2_MODEL_PATH,
            device=device_type
        )
except RuntimeError as e:
    if device_type == "cuda":
        print(f"Failed to initialize CTranslate2 with CUDA: {e}")
        print("Falling back to CPU.")
        device_type = "cpu"
        translator = ctranslate2.Translator(
            CT2_MODEL_PATH,
            device=device_type,
            compute_type=QUANTIZATION,
            intra_threads=os.cpu_count()
        )
    else:
        raise e

print(f"Translator loaded on {device_type}.")
print("-------------------------------------------------------------------------------------------------")

print("Start translation...")

# --- Check for existing output and determine starting point ---
processed_lines = 0
if os.path.exists(OUTPUT_FILE):
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            processed_lines = sum(1 for row in reader if row)
    except FileNotFoundError:
        processed_lines = 0

rows_to_skip = max(0, processed_lines - 1) if processed_lines > 0 else 0

# Count total rows for tqdm
with open(INPUT_FILE, "r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    total_rows = sum(1 for _ in reader)
rows_excluding_header = max(0, total_rows - 1)
remaining_rows = rows_excluding_header - rows_to_skip

print(f"Input CSV `{INPUT_FILE}` total rows: {total_rows}, rows excluding header: {rows_excluding_header}")
if rows_to_skip > 0:
    print(f"Output file `{OUTPUT_FILE}` has {processed_lines} lines. Resuming translation.")
    print(f"Skipping {rows_to_skip} already translated rows.")
    print(f"Remaining rows to translate: {remaining_rows}")
else:
    print("Starting new translation.")

translate_start = time.time()

# --- Open files (append if output exists) ---
output_mode = 'a' if rows_to_skip > 0 else 'w'
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8", newline='') as infile, \
        open(OUTPUT_FILE, output_mode, encoding="utf-8", newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    header = next(reader)  # Read header from input

    if output_mode == 'w':
        new_header = header.copy()
        new_header.append("hu")
        del new_header[1]
        writer.writerow(new_header)
        print("New output file created with header.")

    if rows_to_skip > 0:
        for _ in range(rows_to_skip):
            try:
                next(reader)
            except StopIteration:
                break

    pbar = tqdm(total=rows_excluding_header, initial=rows_to_skip, unit='row', desc='Translating')

    cpu_num = os.cpu_count() or 1
    # Queue for passing batches from main thread (producer) to worker thread (consumer)
    # Each item is a tuple: (batch_rows, source_tokens_list)
    batch_queue = queue.Queue(maxsize=cpu_num * 2)

    def translation_worker():
        while True:
            item = batch_queue.get()
            if item is None:
                batch_queue.task_done()
                break

            batch_rows, source_tokens_list = item
            
            if not batch_rows:
                batch_queue.task_done()
                continue

            try:
                # Translate using CTranslate2 with target prefix for NLLB
                results = translator.translate_batch(
                    source_tokens_list,
                    target_prefix=[[TARGET_LANGUAGE]] * len(source_tokens_list),
                    beam_size=1
                )

                # Decode target sentences
                hu_sentences = [
                    tokenizer.decode(tokenizer.convert_tokens_to_ids(result.hypotheses[0]), skip_special_tokens=True) for
                    result in results]

                for i, original_row in enumerate(batch_rows):
                    hu_sentence = hu_sentences[i]
                    original_row.append(hu_sentence)
                    del original_row[1]
                    writer.writerow(original_row)
                
                # Flush to ensure data is written to disk in case of interruption
                outfile.flush()

                pbar.update(len(batch_rows))
            except Exception as e:
                print(f"Error in translation worker: {e}")
            
            batch_queue.task_done()

    # Start the worker thread
    worker_thread = threading.Thread(target=translation_worker, daemon=True)
    worker_thread.start()

    batch_rows = []
    batch_tokens = []
    current_batch_tokens = 0

    def process_row(row):
        en_sentence = row[0]
        words = en_sentence.split()
        words_count = len(words)
        
        # If sentence is longer than MAX_WORDS_LENGTH, replace with empty string immediately
        if words_count > MAX_WORDS_LENGTH:
            row[0] = ""
            # Tokenize empty string
            tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(""))
        else:
            # Tokenize
            tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(en_sentence))
        return row, tokens

    CHUNK_SIZE = cpu_num * CHUNK_SIZE_MULTIPLIER

    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_num) as executor:
        while True:
            chunk_rows = []
            for _ in range(CHUNK_SIZE):
                try:
                    chunk_rows.append(next(reader))
                except StopIteration:
                    break
            
            if not chunk_rows:
                break

            results = executor.map(process_row, chunk_rows)

            for row, tokens in results:
                num_tokens = len(tokens)

                if batch_rows and (current_batch_tokens + num_tokens > MAX_BATCH_TOKENS):
                    batch_queue.put((batch_rows, batch_tokens))
                    batch_rows = []
                    batch_tokens = []
                    current_batch_tokens = 0

                batch_rows.append(row)
                batch_tokens.append(tokens)
                current_batch_tokens += num_tokens

    # Process remaining rows
    if batch_rows:
        batch_queue.put((batch_rows, batch_tokens))

    # Signal worker to stop
    batch_queue.put(None)
    
    # Wait for worker to finish
    worker_thread.join()

    pbar.close()
    if remaining_rows > 0:
        print(f"\nTranslation appended to: {OUTPUT_FILE}")
    else:
        print(f"\nNo new rows to translate. File is up to date: {OUTPUT_FILE}")

translate_end = time.time()
translate_duration = translate_end - translate_start

print(f"Translation process finished. Total time for this run: {translate_duration:.2f} seconds.")

# --- Cleanup Step ---
print("Starting cleanup of empty rows...")
TEMP_FILE = OUTPUT_FILE + ".tmp"
rows_dropped = 0
rows_kept = 0

try:
    with open(OUTPUT_FILE, "r", encoding="utf-8", newline="") as infile, \
            open(TEMP_FILE, "w", encoding="utf-8", newline="") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        try:
            header = next(reader)
            writer.writerow(header)
        except StopIteration:
            pass  # Empty file

        for row in reader:
            # Check if row is empty or if EN (col 0) or HU (col -1) is empty
            if not row:
                continue

            # Assuming structure [en, ..., hu]
            en_text = row[0].strip() if len(row) > 0 else ""
            hu_text = row[-1].strip() if len(row) > 0 else ""

            if not en_text or not hu_text:
                rows_dropped += 1
                continue

            writer.writerow(row)
            rows_kept += 1

    shutil.move(TEMP_FILE, OUTPUT_FILE)
    print(f"Cleanup complete. Dropped {rows_dropped} rows. Final row count: {rows_kept}.")

except Exception as e:
    print(f"An error occurred during cleanup: {e}")
    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)

print("done")
