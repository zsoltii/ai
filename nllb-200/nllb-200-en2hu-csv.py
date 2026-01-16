import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
import torch, re, csv, time
from tqdm import tqdm
from util.log_available_gpus import log_available_gpus
from util.hfh_login import hfh_login

# Set environment variable to reduce memory fragmentation
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

log_available_gpus()

hfh_login()

# input token number for all models: 512
# MODEL = "facebook/nllb-200-distilled-600M"
MODEL = "facebook/nllb-200-distilled-1.3B"
# MODEL = "facebook/nllb-200-1.3B"
# MODEL = "facebook/nllb-200-3.3B"
#
# max_memory = {
#     0: "14Gib",  # Reduce memory allocation for GPU 0
#     "cpu": "85Gib"
# }

os.makedirs("./offload", exist_ok=True)

# print("Using max_memory config:", max_memory)

# --- Kvantálási Konfiguráció ---
# config = AutoConfig.from_pretrained(MODEL)
#
# if hasattr(config, "quantization_config"):
#     print(f"Original quantization_config {config.quantization_config}")
#     del config.quantization_config
#     print(f"Original quantization_config deleted!")
#
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.half,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_storage=torch.uint8,
#     llm_int8_enable_fp32_cpu_offload=True,
# )

print("Loading tokenizer and model (this may take a while)...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL,
    # config=config,
    # quantization_config=quantization_config,
    dtype=torch.bfloat16,
    device_map="auto",
    # max_memory=max_memory,
    # offload_folder="./offload",
    # low_cpu_mem_usage=False,
    attn_implementation="sdpa",
    # For further speedup on compatible hardware (newer NVIDIA GPUs), uncomment the following line
    # and install flash-attn: pip install flash-attn
    # attn_implementation="flash_attention_2",
)

# --- SPEEDUP: Use torch.compile for a significant performance boost ---
print("Compiling model with torch.compile()... (first run will be slower)")
model = torch.compile(model)

print(f"Model loaded: {getattr(model, 'name_or_path', str(model))}")
print("prepare tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
print("tokenizer prepared")

print("Model loaded. Model device placement:")
print(model.hf_device_map)

print("-------------------------------------------------------------------------------------------------")

print("Start translation...")

SOURCE_LANGUAGE = "eng_Latn"
TARGET_LANGUAGE = "hun_Latn"

tokenizer.src_lang = SOURCE_LANGUAGE

INPUT_FILE = "../text/hu/csv/en-pl.csv"
OUTPUT_FILE =  "../text/hu/csv/en-hu.csv"
BATCH_SIZE = 256 # --- SPEEDUP: Set batch size. Adjust based on your VRAM. ---

# Count rows in the input CSV
with open(INPUT_FILE, "r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    total_rows = sum(1 for _ in reader)
rows_excluding_header = max(0, total_rows - 1)
print(f"Input CSV `{INPUT_FILE}` total rows: {total_rows}, rows excluding header: {rows_excluding_header}")

translate_start = time.time()

with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(OUTPUT_FILE, "w", encoding="utf-8", newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    header = next(reader)
    header.append("hu")
    del header[1]
    writer.writerow(header)

    batch = []
    pbar = tqdm(total=rows_excluding_header, unit='row', desc='Translating')

    for row in reader:
        batch.append(row)

        if len(batch) == BATCH_SIZE:
            en_sentences = [item[0] for item in batch]

            # --- SPEEDUP: Tokenize batch ---
            en_inputs = tokenizer(en_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

            with torch.no_grad():
                outputs = model.generate(**en_inputs,
                                         forced_bos_token_id=tokenizer.convert_tokens_to_ids(TARGET_LANGUAGE),
                                         max_length=512)

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
        en_inputs = tokenizer(en_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**en_inputs,
                                     forced_bos_token_id=tokenizer.convert_tokens_to_ids(TARGET_LANGUAGE),
                                     max_length=512)

        hu_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for i, original_row in enumerate(batch):
            hu_sentence = hu_sentences[i]
            original_row.append(hu_sentence)
            del original_row[1]
            writer.writerow(original_row)

        pbar.update(len(batch))

    pbar.close()
    print(f"\nNew file created: {OUTPUT_FILE}")


translate_end = time.time()
translate_duration = translate_end - translate_start

print(f"Translation completed. Total time: {translate_duration:.2f} seconds.")
print("done")
