import os
import sys
import csv
import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import onnxruntime as ort

MAX_TOKEN_LENGTH = 450 # 512-őt tudo a model kezelni, de legyen egy kis tartalák, hog fordítás után se legyen több

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.hfh_login import hfh_login

# Set environment variable to reduce memory fragmentation
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

hfh_login()

# Determine device and provider
available_providers = ort.get_available_providers()
print(f"Available ONNX providers: {available_providers}")

if "CUDAExecutionProvider" in available_providers:
    provider = "CUDAExecutionProvider"
    device_type = "cuda"
elif "ROCMExecutionProvider" in available_providers:
    provider = "ROCMExecutionProvider"
    device_type = "cuda"
else:
    provider = "CPUExecutionProvider"
    device_type = "cpu"

print(f"Selected provider: {provider} (device: {device_type})")

MODEL_NAME = "Helsinki-NLP/opus-mt-en-hu"
ONNX_MODEL_PATH = "opus-mt-en-hu-onnx"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading ONNX model from {ONNX_MODEL_PATH} (or exporting if needed)...")
model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_NAME, export=True, provider=provider, dtype=torch.bfloat16, provider_options={
    "device_id": 0,
    "arena_extend_strategy": "kSameAsRequested",
})

print(f"Model loaded on {provider}.")
print("-------------------------------------------------------------------------------------------------")

print("Start translation...")

INPUT_FILE = "../text/hu/csv/en-pl.csv"
OUTPUT_FILE = "../text/hu/csv/en-hu-opus-mt-onnx.csv"
BATCH_SIZE = 64

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

    def process_batch(batch_rows):
        if not batch_rows:
            return

        en_sentences = [item[0] for item in batch_rows]
        
        # Filter by length
        encoded = tokenizer(en_sentences, truncation=False, padding=False)
        valid_indices = []
        valid_en_sentences = []
        hu_sentences = [""] * len(batch_rows)

        for i, ids in enumerate(encoded.input_ids):
            if len(ids) <= MAX_TOKEN_LENGTH:
                valid_indices.append(i)
                valid_en_sentences.append(en_sentences[i])
        
        if valid_en_sentences:
            # Tokenize for model
            inputs = tokenizer(valid_en_sentences, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKEN_LENGTH)
            
            # Generate
            outputs = model.generate(**inputs)
            
            # Decode
            decoded_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for i, sent in zip(valid_indices, decoded_sentences):
                hu_sentences[i] = sent

        for i, original_row in enumerate(batch_rows):
            hu_sentence = hu_sentences[i]
            original_row.append(hu_sentence)
            del original_row[1]
            writer.writerow(original_row)

    for row in reader:
        batch.append(row)

        if len(batch) == BATCH_SIZE:
            process_batch(batch)
            pbar.update(len(batch))
            batch = []

    if batch:
        process_batch(batch)
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
