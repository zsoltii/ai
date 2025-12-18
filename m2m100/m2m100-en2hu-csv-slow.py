import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, AutoConfig, BitsAndBytesConfig
import torch, re, csv
from tqdm import tqdm
from util.log_available_gpus import log_available_gpus
from util.hfh_login import hfh_login

# Set environment variable to reduce memory fragmentation
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

log_available_gpus()

hfh_login()

# input token number for all models: 512
MODEL = "facebook/m2m100_1.2B"
# MODEL = "NYTK/translation-m2m100-1.2B-multi12-hungarian" # input token number 256!!!
# MODEL = "facebook/m2m100_418M"
# MODEL = "facebook/m2m100-12B-last-ckpt"
# MODEL = "facebook/m2m100-12B-avg-5-ckpt"
# MODEL = "facebook/m2m100-12B-avg-10-ckpt"

max_memory = {
    0: "14Gib",  # Reduce memory allocation for GPU 0
    "cpu": "85Gib"
}

os.makedirs("./offload", exist_ok=True)

print("Using max_memory config:", max_memory)

# --- Kvantálási Konfiguráció ---
# A modell eredeti kvantálási sémájának felülbírálása.
config = AutoConfig.from_pretrained(MODEL)

# if hasattr(config, "quantization_config"):
#     print(f"Original quantization_config {config.quantization_config}")
#     del config.quantization_config
#     print(f"Original quantization_config deleted!")
#
# # Létrehozunk egy új, 4-bites kvantálási konfigurációt, amely támogatja a CPU offload-ot.
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.half,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_storage=torch.uint8,
#     llm_int8_enable_fp32_cpu_offload=True,  # Engedélyezi a CPU-ra történő offload-ot
# )

print("Loading tokenizer and model (this may take a while)...")

model = M2M100ForConditionalGeneration.from_pretrained(
    MODEL,
    config=config,
    # quantization_config=quantization_config,
    dtype=torch.float32,
    device_map="auto",
    max_memory=max_memory,
    offload_folder="./offload",
    low_cpu_mem_usage=False,
)

print(f"Model loaded: {getattr(model, 'name_or_path', str(model))}")

print(f"Model loaded: {getattr(model, 'name_or_path', str(model))}")
print("prepare tokenizer...")
tokenizer = M2M100Tokenizer.from_pretrained(MODEL)
print("tokenizer prepared")

print("Model loaded. Model device placement:")
print(model.hf_device_map)

print("-------------------------------------------------------------------------------------------------")

print("Start translation...")

SOURCE_LANGUAGE = "en"
TARGET_LANGUAGE = "hu"

tokenizer.src_lang = SOURCE_LANGUAGE

source_file = "../text/en/The_wonderful_wizard_of_Oz.txt"
with open(source_file, "r", encoding="utf-8") as file:
    source_text = file.read()

# Generate output filename: in the same folder as the source_file
source_dir = os.path.dirname(source_file)
source_base = os.path.basename(source_file)
base_name, ext = os.path.splitext(source_base)
output_file = os.path.join(source_dir, f"{base_name}.hu{ext}")
# Clear the output file before translation if it exists
open(output_file, "w", encoding="utf-8").close()

# Group lines into paragraphs, where paragraphs are separated by empty lines.
# This handles sentences that span multiple lines.
source_text_lines = source_text.split('\n')
paragraphs = []
current_paragraph = ""
for line in source_text_lines:
    # An empty line signifies a paragraph break.
    if not line.strip():
        if current_paragraph:
            paragraphs.append(current_paragraph)
            current_paragraph = ""
        # Preserve the empty line to maintain paragraph separation in the output.
        paragraphs.append("")
    else:
        # Append the line to the current paragraph.
        if current_paragraph:
            # Add a newline to separate from the previous line within the same paragraph.
            current_paragraph += "\n" + line
        else:
            current_paragraph = line
# Add the last paragraph if the text doesn't end with an empty line.
if current_paragraph:
    paragraphs.append(current_paragraph)

import time

translate_start = time.time()

# Calculate total number of sentences for progress tracking
total_sentences = 0
for para in paragraphs:
    if para.strip():
        total_sentences += len(re.split(r'(?<=[.!?])\s+', para))

sentences_processed = 0

# Initialize variables to track the maximum token counts
max_input_tokens = 0
max_generated_tokens = 0

INPUT_FILE = "../text/hu/csv/en-pl.csv"
OUTPUT_FILE =  "../text/hu/csv/en-hu.csv"

# Count rows in the input CSV and print the total (and excluding header)
with open(INPUT_FILE, "r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    total_rows = sum(1 for _ in reader)
rows_excluding_header = max(0, total_rows - 1)
print(f"Input CSV `{INPUT_FILE}` total rows: {total_rows}, rows excluding header: {rows_excluding_header}")

with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(OUTPUT_FILE, "w", encoding="utf-8", newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    header = next(reader)
    header.append("hu")
    del header[1] # delete polish column in the line
    writer.writerow(header)

    pbar = tqdm(reader, total=rows_excluding_header, unit='row', desc='Translating')

    for row_number, row in enumerate(pbar, 1):
        en_sentence = row[0]

        tqdm.write(f"Translating sentence [{SOURCE_LANGUAGE}] ({row_number}/{rows_excluding_header}): {en_sentence}")

        en_inputs = tokenizer(en_sentence, return_tensors="pt").to(model.device)
        num_input_tokens = en_inputs.input_ids.shape[1]
        tqdm.write(f"Number of tokens: {num_input_tokens}")
        max_input_tokens = max(max_input_tokens, num_input_tokens)

        with torch.no_grad():
            m2m100_outputs = model.generate(**en_inputs,
                                            forced_bos_token_id=tokenizer.get_lang_id(TARGET_LANGUAGE),
                                            max_length=512)
        num_generated_tokens = m2m100_outputs.shape[1]
        tqdm.write(f"Number of generated tokens: {num_generated_tokens}")
        max_generated_tokens = max(max_generated_tokens, num_generated_tokens)

        hu_sentence = tokenizer.batch_decode(m2m100_outputs, skip_special_tokens=True)[0]  # this line runs on CPU!

        row.append(hu_sentence)

        tqdm.write(f"Translated sentence [{TARGET_LANGUAGE}]: {hu_sentence}")

        del row[1]  # delete polish column in the line

        writer.writerow(row)

        # update the bar postfix with token info
        pbar.set_postfix({'in': num_input_tokens, 'out': num_generated_tokens, 'max_in': max_input_tokens, 'max_out': max_generated_tokens})

    print(f"new file: {OUTPUT_FILE}")

# Print the maximum token counts after processing
print(f"Maximum number of input tokens: {max_input_tokens}")
print(f"Maximum number of generated tokens: {max_generated_tokens}")

translate_end = time.time()
translate_duration = translate_end - translate_start

print("Translation completed. Output saved to:", output_file)
print(f"Translation time: {translate_duration:.2f} seconds (translation only)")
print("done")
