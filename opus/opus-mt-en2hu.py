import os
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Detect ROCm/AMD GPU and set device accordingly
if torch.cuda.is_available() and torch.version.hip:
    device = torch.device('cuda')  # ROCm uses 'cuda' as device string
    print('ROCm detected, using device:', device)
else:
    device = torch.device('cpu')
    print('No ROCm/GPU detected, using CPU')

# A Helsinki-NLP/opus-mt-en-hu modell és tokenizer betöltése,
# ami kifejezetten angol-magyar fordításra lett tanítva.
model_name = 'Helsinki-NLP/opus-mt-en-hu'
print("load model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
).to(device)
print(f"Model loaded: {model_name}")
print("-----------------------------------------------------")

# A fordítandó szöveg
source_file = "../text/en/mig-29.txt"
with open(source_file, "r", encoding="utf-8") as file:
    source_text = file.read()

# Generate output filename: in the same folder as the source_file
source_dir = os.path.dirname(source_file)
source_base = os.path.basename(source_file)
base_name, ext = os.path.splitext(source_base)
output_file = os.path.join(source_dir, f"{base_name}.hu.opus{ext}")
# Clear the output file before translation if it exists
open(output_file, "w", encoding="utf-8").close()

# Split text by lines to treat each line as a paragraph
paragraphs = source_text.split('\n')

translate_start = time.time()

# Calculate total number of sentences for progress tracking
total_sentences = 0
for para in paragraphs:
    if para.strip():
        total_sentences += len(re.split(r'(?<=[.!?])\s+', para))

sentences_processed = 0

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
            # A szöveg tokenizálása
            input_ids = tokenizer.encode(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

            # A fordítás végrehajtása
            with torch.inference_mode():
                outputs = model.generate(**input_ids,
                                         max_new_tokens=256,
                                         num_beams=1,  # Gyorsabb, mint a beam search
                                         do_sample=False,
                                         use_cache=True,
                                         )
                translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
                translated_sentences.append(translation)

            sentences_processed += 1
            percent = (sentences_processed / total_sentences) * 100 if total_sentences > 0 else 0
            print(f"Progress: {percent:.1f}% ({sentences_processed}/{total_sentences})")

        translated_paragraph = " ".join(translated_sentences)
        out_f.write(translated_paragraph + "\n")


translate_end = time.time()
translate_duration = translate_end - translate_start

print("Translation completed. Output saved to:", output_file)
print(f"Translation time: {translate_duration:.2f} seconds (translation only)")
print("done")
