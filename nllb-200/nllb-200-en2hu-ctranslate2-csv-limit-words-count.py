import os
import sys
import csv
import time
import shutil
import torch
from tqdm import tqdm
import ctranslate2
from transformers import AutoTokenizer
from ctranslate2.converters import TransformersConverter

# NLLB-200 specific configuration
MAX_WORDS_LENGTH = 450

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

# NLLB Model Configuration
MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
CT2_MODEL_PATH = "nllb-200-distilled-1.3B-ct2"
SOURCE_LANGUAGE = "eng_Latn"
TARGET_LANGUAGE = "hun_Latn"

# Check if conversion is needed
quantization = "int8"

if not os.path.exists(CT2_MODEL_PATH):
    print(f"Converting model {MODEL_NAME} to CTranslate2 format...")
    converter = TransformersConverter(MODEL_NAME)
    converter.convert(output_dir=CT2_MODEL_PATH, quantization=quantization, force=True)
    print(f"Model converted to {CT2_MODEL_PATH} with quantization {quantization}.")
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
            compute_type=quantization,
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
            compute_type=quantization,
            intra_threads=os.cpu_count()
        )
    else:
        raise e

print(f"Translator loaded on {device_type}.")
print("-------------------------------------------------------------------------------------------------")

print("Start translation...")

INPUT_FILE = "../text/hu/csv/en-pl.csv"
OUTPUT_FILE = "../text/hu/csv/en-hu-nllb-200-ct2.csv"
MAX_BATCH_WORDS = 3000

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

    batch = []
    current_batch_words = 0
    pbar = tqdm(total=rows_excluding_header, initial=rows_to_skip, unit='row', desc='Translating')


    def process_batch(batch_rows):
        if not batch_rows:
            return

        en_sentences = [item[0] for item in batch_rows]
        
        # Tokenize source sentences
        source_tokens_list = [tokenizer.convert_ids_to_tokens(tokenizer.encode(s)) for s in en_sentences]
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

        pbar.update(len(batch_rows))


    for row in reader:
        en_sentence = row[0]
        words = en_sentence.split()
        words_count = len(words)
        
        # If sentence is longer than MAX_WORDS_LENGTH, replace with empty string immediately
        # to save memory and skip processing in process_batch.
        if words_count > MAX_WORDS_LENGTH:
            row[0] = ""
            effective_words = 0
        else:
            effective_words = words_count

        if batch and (current_batch_words + effective_words > MAX_BATCH_WORDS):
            process_batch(batch)
            batch = []
            current_batch_words = 0

        batch.append(row)
        current_batch_words += effective_words

    # Process remaining rows
    process_batch(batch)

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
