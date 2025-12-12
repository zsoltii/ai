import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, AutoConfig, BitsAndBytesConfig
import torch
import re
from util.select_device import select_device
from util.log_available_gpus import log_available_gpus
from util.hfh_login import hfh_login

# Set environment variable to reduce memory fragmentation
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

log_available_gpus()
device = select_device()

hfh_login()

# input token number for all models: 512
# MODEL = "facebook/m2m100_1.2B"
# MODEL = "NYTK/translation-m2m100-1.2B-multi12-hungarian" # input token number 256!!!
# MODEL = "facebook/m2m100_418M"
# MODEL = "facebook/m2m100-12B-last-ckpt"
MODEL = "facebook/m2m100-12B-avg-5-ckpt"
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

if hasattr(config, "quantization_config"):
    print(f"Original quantization_config {config.quantization_config}")
    del config.quantization_config
    print(f"Original quantization_config deleted!")

# Létrehozunk egy új, 4-bites kvantálási konfigurációt, amely támogatja a CPU offload-ot.
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.half,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.uint8,
    llm_int8_enable_fp32_cpu_offload=True,  # Engedélyezi a CPU-ra történő offload-ot
)

print("Loading tokenizer and model (this may take a while)...")

model = M2M100ForConditionalGeneration.from_pretrained(
    MODEL,
    config=config,
    quantization_config=quantization_config,
    dtype=torch.half,
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

source_file = "../text/en/mig-29.txt"
with open(source_file, "r", encoding="utf-8") as file:
    source_text = file.read()

# Generate output filename: in the same folder as the source_file
source_dir = os.path.dirname(source_file)
source_base = os.path.basename(source_file)
base_name, ext = os.path.splitext(source_base)
output_file = os.path.join(source_dir, f"{base_name}.hu{ext}")
# Clear the output file before translation if it exists
open(output_file, "w", encoding="utf-8").close()

# Split text by lines to treat each line as a paragraph
paragraphs = source_text.split('\n')

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

with open(output_file, "a", encoding="utf-8") as out_f:
    for para in paragraphs:
        if not para.strip():
            out_f.write("\n")
            continue

        # Split by sentence-ending punctuation (., !, ?) followed by space or end of line
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', para) if s.strip()]
        if not sentences:
            continue

        translated_sentences = []
        for sentence in sentences:
            print(f"Translating sentence: {sentence}")

            encoded_hi = tokenizer(sentence, return_tensors="pt").to(device)
            num_input_tokens = encoded_hi.input_ids.shape[1]
            print(f"Number of tokens: {num_input_tokens}")
            max_input_tokens = max(max_input_tokens, num_input_tokens)

            with torch.no_grad():
                generated_tokens = model.generate(**encoded_hi,
                                                  forced_bos_token_id=tokenizer.get_lang_id(TARGET_LANGUAGE))
                num_generated_tokens = generated_tokens.shape[1]
                print(f"Number of generated tokens: {num_generated_tokens}")
                max_generated_tokens = max(max_generated_tokens, num_generated_tokens)

                translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[
                    0]  # this line runs on CPU!
                print(f"Translation: {translation}")
                translated_sentences.append(translation)

            sentences_processed += 1
            percent = (sentences_processed / total_sentences) * 100 if total_sentences > 0 else 0
            print(f"Progress: {percent:.1f}% ({sentences_processed}/{total_sentences})")

        translated_paragraph = " ".join(translated_sentences)
        out_f.write(translated_paragraph + "\n")

# Print the maximum token counts after processing
print(f"Maximum number of input tokens: {max_input_tokens}")
print(f"Maximum number of generated tokens: {max_generated_tokens}")

translate_end = time.time()
translate_duration = translate_end - translate_start

print("Translation completed. Output saved to:", output_file)
print(f"Translation time: {translate_duration:.2f} seconds (translation only)")
print("done")
