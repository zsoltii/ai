import os
import sys
import csv
import time
import torch
from tqdm import tqdm
import ctranslate2
from transformers import AutoTokenizer
from ctranslate2.converters import TransformersConverter
MAX_TOKEN_LENGTH = 450 # 512-őt tudo a model kezelni, de legyen egy kis tartalák, hog fordítás után se legyen több


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.log_available_gpus import log_available_gpus
from util.select_device import select_device
from util.hfh_login import hfh_login

# Set environment variable to reduce memory fragmentation
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# log_available_gpus()
# device_obj = select_device()
hfh_login()

# Determine device for CTranslate2
# Check if CTranslate2 sees any CUDA devices (standard builds might not see AMD ROCm)
try:
    cuda_count = ctranslate2.get_cuda_device_count()
except Exception:
    cuda_count = 0

device_type = "cuda" if cuda_count > 0 else "cpu"
print(f"CTranslate2 detected {cuda_count} CUDA devices. Initial selection: {device_type}")

MODEL_NAME = "Helsinki-NLP/opus-mt-en-hu"
CT2_MODEL_PATH = "opus-mt-en-hu-ct2"

# Check if conversion is needed
# We use 'int8' quantization because it is supported on both CPU and GPU.
# This ensures that if we fallback to CPU, the model is still compatible.
quantization = "int8"

if not os.path.exists(CT2_MODEL_PATH):
    print(f"Converting model {MODEL_NAME} to CTranslate2 format...")
    converter = TransformersConverter(MODEL_NAME)
    converter.convert(output_dir=CT2_MODEL_PATH, quantization=quantization, force=True)
    # converter.convert(output_dir=CT2_MODEL_PATH, force=True)
    print(f"Model converted to {CT2_MODEL_PATH} with quantization {quantization}.")
else:
    # Optional: Check if we should warn about quantization if it was converted differently before
    pass

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

print(f"Loading CTranslate2 translator on {device_type}...")
try:
    if device_type == "cpu":
        translator = ctranslate2.Translator(
            CT2_MODEL_PATH,
            device=device_type,
            compute_type=quantization,
            inter_threads=64,     # Párhuzamosan feldolgozott streamek száma
            intra_threads=os.cpu_count()     # Egy-egy számítás során használt CPU szálak száma
        )
    else:
        translator = ctranslate2.Translator(
            CT2_MODEL_PATH,
            device=device_type
    )
except RuntimeError as e:
    if device_type == "cuda":
        print(f"Failed to initialize CTranslate2 with CUDA: {e}")
        print("This often happens if CTranslate2 is not compiled with ROCm support on AMD systems.")
        print("Falling back to CPU. Note: This might be slower.")
        device_type = "cpu"
        translator = ctranslate2.Translator(
            CT2_MODEL_PATH,
            device=device_type,
            compute_type=quantization,
            # inter_threads=48,     # Párhuzamosan feldolgozott streamek száma
            intra_threads=os.cpu_count()     # Egy-egy számítás során használt CPU szálak száma
        )
    else:
        raise e

print(f"Translator loaded on {device_type}.")
print("-------------------------------------------------------------------------------------------------")

print("Start translation...")

INPUT_FILE = "../text/hu/csv/en-pl.csv"
OUTPUT_FILE = "../text/hu/csv/en-hu-opus-mt.csv"
BATCH_SIZE = 128

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
    pbar = tqdm(total=rows_excluding_header, initial=rows_to_skip, unit='row', desc='Translating')

    for row in reader:
        batch.append(row)

        if len(batch) == BATCH_SIZE:
            en_sentences = [item[0] for item in batch]

            # Tokenize
            source_tokens_list = [tokenizer.convert_ids_to_tokens(tokenizer.encode(text)) for text in en_sentences]

            valid_indices = []
            valid_source_tokens = []
            hu_sentences = [""] * len(batch)

            for i, tokens in enumerate(source_tokens_list):
                if len(tokens) <= MAX_TOKEN_LENGTH:
                    valid_indices.append(i)
                    valid_source_tokens.append(tokens)
                # else: hu_sentences[i] remains ""

            if valid_source_tokens:
                # Translate
                results = translator.translate_batch(valid_source_tokens)

                # Decode
                decoded_sentences = [tokenizer.decode(tokenizer.convert_tokens_to_ids(result.hypotheses[0])) for result in results]
                
                for i, sent in zip(valid_indices, decoded_sentences):
                    hu_sentences[i] = sent

            for i, original_row in enumerate(batch):
                hu_sentence = hu_sentences[i]
                original_row.append(hu_sentence)
                del original_row[1]
                writer.writerow(original_row)

            pbar.update(len(batch))
            batch = []

    if batch:
        en_sentences = [item[0] for item in batch]
        
        # Tokenize
        source_tokens_list = [tokenizer.convert_ids_to_tokens(tokenizer.encode(text)) for text in en_sentences]

        valid_indices = []
        valid_source_tokens = []
        hu_sentences = [""] * len(batch)

        for i, tokens in enumerate(source_tokens_list):
            if len(tokens) <= MAX_TOKEN_LENGTH:
                valid_indices.append(i)
                valid_source_tokens.append(tokens)
            # else: hu_sentences[i] remains ""

        if valid_source_tokens:
            # Translate
            results = translator.translate_batch(valid_source_tokens)

            # Decode
            decoded_sentences = [tokenizer.decode(tokenizer.convert_tokens_to_ids(result.hypotheses[0])) for result in results]
            
            for i, sent in zip(valid_indices, decoded_sentences):
                hu_sentences[i] = sent

        for i, original_row in enumerate(batch):
            hu_sentence = hu_sentences[i]
            original_row.append(hu_sentence)
            del original_row[1]
            writer.writerow(original_row)

        pbar.update(len(batch))

    pbar.close()
    if remaining_rows > 0:
        print(f"\nTranslation appended to: {OUTPUT_FILE}")
    else:
        print(f"\nNo new rows to translate. File is up to date: {OUTPUT_FILE}")

translate_end = time.time()
translate_duration = translate_end - translate_start

print(f"Translation process finished. Total time for this run: {translate_duration:.2f} seconds.")
print("done")
