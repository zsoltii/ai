import os
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
import re

# Detect ROCm/AMD GPU and set device accordingly
if torch.version.hip is not None:
    device = torch.device('cuda')  # ROCm uses 'cuda' as device string
    print('ROCm detected, using device:', device)
else:
    device = torch.device('cpu')
    print('No ROCm detected, using CPU')

print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(f'device name [0]:', torch.cuda.get_device_name(0))
print(torch.version.cuda)
print('torch.version.hip:', getattr(torch.version, 'hip', None))

MODEL = "facebook/m2m100_1.2B"
#MODEL = "facebook/m2m100_418M"
#MODEL = "facebook/m2m100-12B-last-ckpt"

SOURCE_LANGUAGE="hu"
TARGET_LANGUAGE="en"

with open("../text/1.txt", "r", encoding="utf-8") as file:
    source_text = file.read()

# Split text into chunks (by paragraphs and by sentences)
paragraphs = [p.strip() for p in source_text.split('\n\n') if p.strip()]
sentence_chunks = []
for para in paragraphs:
    # Split by sentence-ending punctuation (., !, ?) followed by space or end of line
    sentences = re.split(r'(?<=[.!?])\s+', para)
    sentence_chunks.extend([s.strip() for s in sentences if s.strip()])
chunks = sentence_chunks

print("load model...")
model = M2M100ForConditionalGeneration.from_pretrained(MODEL)
print(f"Model loaded: {getattr(model, 'name_or_path', str(model))}")
model = model.to(device)
print("prepare tokenizer...")
tokenizer = M2M100Tokenizer.from_pretrained(MODEL)
print("-----------------------------------------------------")
tokenizer.src_lang = SOURCE_LANGUAGE

translations = []
for idx, chunk in enumerate(chunks):
    encoded_hi = tokenizer(chunk, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id(TARGET_LANGUAGE))
        hu_to_en = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translations.append(hu_to_en[0])
    percent = ((idx + 1) / len(chunks)) * 100
    print(f"Progress: {percent:.1f}% ({idx + 1}/{len(chunks)})")

full_translation = '\n\n'.join(translations)
print(full_translation)
print("-----------------------------------------------------")
print("Hungarian to English:", full_translation)
