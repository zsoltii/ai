import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch, csv, time
from tqdm import tqdm
from util.log_available_gpus import log_available_gpus
from util.select_device import select_device
from util.hfh_login import hfh_login

# Set environment variable to reduce memory fragmentation
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

log_available_gpus()
device = select_device()
hfh_login()

# input token number for all models: 512
MODEL = "Helsinki-NLP/opus-mt-en-hu"

print("Loading tokenizer and model (this may take a while)...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
).to(device)

# --- SPEEDUP: Use torch.compile for a significant performance boost ---
print("Compiling model with torch.compile()... (first run will be slower)")
model = torch.compile(model, mode="reduce-overhead")

print(f"Model loaded: {getattr(model, 'name_or_path', str(model))}")
print("prepare tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
print("tokenizer prepared")

if hasattr(model, "hf_device_map"):
    print("Model loaded. Model device placement:")
    print(model.hf_device_map)

print("-------------------------------------------------------------------------------------------------")

print("Start translation...")

INPUT_FILE = "../text/hu/csv/en-pl.csv"
OUTPUT_FILE = "../text/hu/csv/en-hu-opus-mt.csv"
BATCH_SIZE = 96  # --- SPEEDUP: Set batch size. Adjust based on your VRAM. ---

# --- Check for existing output and determine starting point ---
processed_lines = 0
if os.path.exists(OUTPUT_FILE):
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8', newline='') as f:
            # Use csv.reader to handle potential empty lines better
            reader = csv.reader(f)
            processed_lines = sum(1 for row in reader if row)  # Count non-empty rows
    except FileNotFoundError:
        processed_lines = 0

# If file is not empty, we have a header + processed rows
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
# Ensure the directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8", newline='') as infile, \
        open(OUTPUT_FILE, output_mode, encoding="utf-8", newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    header = next(reader)  # Read header from input

    # --- Write header only if creating a new file ---
    if output_mode == 'w':
        new_header = header.copy()
        new_header.append("hu")
        del new_header[1]
        writer.writerow(new_header)
        print("New output file created with header.")

    # --- Skip already processed rows in the input file ---
    if rows_to_skip > 0:
        for _ in range(rows_to_skip):
            try:
                next(reader)
            except StopIteration:
                break  # Reached end of file, nothing to do

    batch = []
    # --- Adjust tqdm total and initial progress ---
    pbar = tqdm(total=rows_excluding_header, initial=rows_to_skip, unit='row', desc='Translating')

    for row in reader:
        batch.append(row)

        if len(batch) == BATCH_SIZE:
            en_sentences = [item[0] for item in batch]

            # --- SPEEDUP: Tokenize batch ---
            en_inputs = tokenizer(en_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                model.device)

            with torch.inference_mode():
                outputs = model.generate(**en_inputs,
                                         max_new_tokens=256,
                                         num_beams=1,  # Gyorsabb, mint a beam search
                                         do_sample=False,
                                         use_cache=True,
                                         )

            # --- SPEEDUP: Decode batch ---
            hu_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for i, original_row in enumerate(batch):
                hu_sentence = hu_sentences[i]
                original_row.append(hu_sentence)
                del original_row[1]
                writer.writerow(original_row)

            pbar.update(len(batch))
            batch = []

    # --- Process the final, smaller batch ---
    if batch:
        en_sentences = [item[0] for item in batch]
        en_inputs = tokenizer(en_sentences, return_tensors="pt", padding=True, truncation=True, max_length=256).to(
            model.device)

        with torch.no_grad():
            outputs = model.generate(**en_inputs, max_length=256)

        hu_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True)

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
