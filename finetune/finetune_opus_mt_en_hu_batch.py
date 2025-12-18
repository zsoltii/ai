import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math, torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig, AutoModelForSeq2SeqLM
)
from peft import get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig

from util.log_available_gpus import log_available_gpus
from util.hfh_login import hfh_login
from util.finetune import should_resume_from_checkpoint, create_new_model_name, get_last_checkpoint, \
    QUANTIZATION_CONFIG, RESULTS_DIRECTORY, PEFT_CONFIG

hfh_login()
log_available_gpus()

# --- Modell és Tokenizer beállítások ---
BASE_MODEL_ID = "microsoft/Phi-4-mini-reasoning"
# A finomhangolt modell mentési neve (LoRA adapter)
NEW_MODEL_NAME = create_new_model_name(BASE_MODEL_ID, "finetuned")
# A finomhangolt LoRA adapter könyvtára
ADAPTER_MODEL_PATH = "./" + NEW_MODEL_NAME

SENTENCE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
SENTENCE_COUNT = 48
GENERATION_BATCH_SIZE = 4
SENTENCE_GENERATE_MESSAGES = [
    {"role": "system", "content": "You are a creative linguistic expert. Output ONLY one grammatically correct English sentence. You must vary the sentence types across the entire English spectrum: declarative, interrogative, imperative, exclamatory, conditional (if/then), optative (wishes), or hypothetical. Use diverse structures including passive voice, compound-complex forms, and varied tenses. Do not use any internal reasoning or preamble. Output ONLY the raw text of the sentence between 3 and 100 words."},
    {"role": "user", "content": "Generate one unique, complex, or creative English sentence. Use any sentence type (statement, question, command, exclamation, conditional, or wish). No thinking, no explain, just the sentence."}
]

PER_DEVICE_TRAIN_BATCH_SIZE = 10
GRADIENT_ACCUMULATION_STEPS = 4
# NUM_EPOCHS = 10 # ez az ideális, kb 90%-os pontosság érhető el vele, viszont 13-14 nap a magyar wikipediát feldolgozni egy AMD 6900 XT-vel
NUM_EPOCHS = 10

# --- Opus-MT beállítások ---
OPUS_MODEL_ID = "Helsinki-NLP/opus-mt-en-hu"

# --- Memória és Offload beállítások ---
MAX_MEMORY = {
    0: "14Gib",  # Memória korlát a 0-s GPU-ra
    "cpu": "85Gib"  # Memória korlát a CPU-ra (offload esetén)
}
os.makedirs("./offload_m2m100", exist_ok=True)
print("max_memory konfiguráció használata:", MAX_MEMORY)


# --- Adathalmaz generálása (in-memory) ---
def generate_translation_dataset(sentence_count, batch_size):
    data = []

    # -- opus betöltése ---
    print(f"'{OPUS_MODEL_ID}' modell betöltése...")
    opus_model = AutoModelForSeq2SeqLM.from_pretrained(
        OPUS_MODEL_ID,
        dtype=torch.float32,
        device_map="auto",
        max_memory=MAX_MEMORY,
        offload_folder="./offload",
        low_cpu_mem_usage=False,
        attn_implementation="sdpa",
    )
    print("Compiling opus_model with torch.compile()... (first run will be slower)")
    opus_model = torch.compile(opus_model)
    opus_tokenizer = AutoTokenizer.from_pretrained(OPUS_MODEL_ID)
    print("\nOpus-MT Modell betöltve. A modell elhelyezkedése:")
    print(opus_model.hf_device_map)

    # --- mondat generáló model betöltése ---
    print(f"'{SENTENCE_MODEL_ID}' modell betöltése...")
    sentence_model = AutoModelForCausalLM.from_pretrained(
        SENTENCE_MODEL_ID,
        device_map="auto",
        max_memory=MAX_MEMORY,
        dtype=torch.half,
        low_cpu_mem_usage=False,
        offload_folder="./offload_sentence",
    )
    sentence_model = torch.compile(sentence_model)
    sentence_tokenizer = AutoTokenizer.from_pretrained(SENTENCE_MODEL_ID)
    # A CausalLM modellekhez a padding token beállítása elengedhetetlen a batch generáláshoz
    if sentence_tokenizer.pad_token is None:
        sentence_tokenizer.pad_token = sentence_tokenizer.eos_token
    sentence_tokenizer.padding_side = "left"  # Left padding for decoder-only models

    print(f"\n{SENTENCE_MODEL_ID} Modell betöltve. A modell elhelyezkedése:")
    print(sentence_model.hf_device_map)

    # --- mondatok generálása batch-ekben ---
    num_batches = math.ceil(sentence_count / batch_size)
    with tqdm(total=sentence_count, desc="Adathalmaz generálása", unit="mondat") as pbar:
        for i in range(num_batches):
            current_batch_size = min(batch_size, sentence_count - len(data))
            if current_batch_size <= 0:
                break

            pbar.set_description(f"Batch {i + 1}/{num_batches}: Angol mondatok generálása")
            batch_input_texts = [
                sentence_tokenizer.apply_chat_template(SENTENCE_GENERATE_MESSAGES, tokenize=False,
                                                       add_generation_prompt=True)
                for _ in range(current_batch_size)
            ]
            en_sentence_inputs = sentence_tokenizer(batch_input_texts, return_tensors="pt", padding=True).to(
                sentence_model.device)

            with torch.no_grad():
                sentence_outputs = sentence_model.generate(
                    **en_sentence_inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=1.2,
                    top_p=0.95,
                    top_k=60,
                    repetition_penalty=1.15,
                    pad_token_id=sentence_tokenizer.eos_token_id,
                    num_return_sequences=1
                )

            generated_ids = sentence_outputs[:, en_sentence_inputs.input_ids.shape[1]:]
            decoded_sentences = sentence_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            en_sentences = []
            for sentence in decoded_sentences:
                clean_sentence = sentence.strip()
                if "(" in clean_sentence:
                    clean_sentence = clean_sentence.partition("(")[0].strip()
                en_sentences.append(clean_sentence)

            pbar.set_description(f"Batch {i + 1}/{num_batches}: Fordítás magyarra")
            opus_tokenizer.padding_side = "right"
            opus_inputs = opus_tokenizer(en_sentences, return_tensors="pt", padding=True, truncation=True).to(
                opus_model.device)

            with torch.no_grad():
                opus_outputs = opus_model.generate(**opus_inputs, max_length=1024)

            hu_sentences = opus_tokenizer.batch_decode(opus_outputs, skip_special_tokens=True)

            for en_sentence, hu_sentence in zip(en_sentences, hu_sentences):
                structured_text = (
                    "instruction: Translate the following English sentence to Hungarian\n"
                    f"input: {en_sentence}\n"
                    f"output: {hu_sentence}"
                )
                data.append({"text": structured_text})
            pbar.update(current_batch_size)

    del opus_model, opus_tokenizer, sentence_model, sentence_tokenizer
    torch.cuda.empty_cache()

    return data


print("Adathalmaz generálása...")
translation_data = generate_translation_dataset(SENTENCE_COUNT, GENERATION_BATCH_SIZE)
dataset = Dataset.from_list(translation_data)
print(f"In-memory adathalmaz létrehozva. Méret: {dataset.num_rows}")

print("Dokumentumok számának meghatározása...")
num_documents = dataset.num_rows
if num_documents == 0:
    raise ValueError("Nem található feldolgozható adat a megadott könyvtárban.")
print(f"Talált dokumentumok száma: {num_documents}")

# --- Modell betöltése ---
print(f"'{BASE_MODEL_ID}' modell betöltése 4-bites kvantálással...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=QUANTIZATION_CONFIG,
    device_map="auto",
    max_memory=MAX_MEMORY,
    dtype=torch.half,
    low_cpu_mem_usage=False,
    offload_folder="./offload",
    trust_remote_code=True,
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# --- Alapértelmezett Tokenizer betöltése ---
print("Alapértelmezett Tokenizer betöltése")
base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
base_tokenizer.pad_token = base_tokenizer.eos_token
base_tokenizer.padding_side = "right"

# --- Teljes adathalmaz finomhangolása ---
output_dir = RESULTS_DIRECTORY + "-" + NEW_MODEL_NAME
print(f"--- Eredmények mentése ide: {output_dir} ---")

model = base_model
tokenizer = base_tokenizer

adapter_path = os.path.join(ADAPTER_MODEL_PATH, "adapter_model.safetensors")
is_adapter_saved = os.path.exists(adapter_path)
last_checkpoint = get_last_checkpoint(output_dir)

if last_checkpoint:
    checkpoint_path = os.path.join(output_dir, last_checkpoint)
    print(f"Meglévő adapter és toknaizer betöltése a '{checkpoint_path}' könyvtárból a tanítás folytatásához...")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    print("Adapter sikeresen betöltve.")
elif is_adapter_saved:
    print(f"Meglévő adapter és tokanizer betöltése a '{ADAPTER_MODEL_PATH}' könyvtárból a tanítás folytatásához...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL_PATH, trust_remote_code=True)
    print("Adapter sikeresen betöltve.")
else:
    print("Nem található meglévő adapter. Új adapter létrehozása...")
    model = get_peft_model(base_model, PEFT_CONFIG)
    print("Új adapter sikeresen létrehozva.")
    print("A modell felkészítve a PEFT (LoRA) tanításra.")

print("\nModell betöltve. A modell elhelyezkedése:")
print(model.hf_device_map)

# --- Tanítási Argumentumok (Dinamikus lépésszámmal) ---
effective_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
steps_per_epoch = math.ceil(num_documents / effective_batch_size)
max_steps = steps_per_epoch * NUM_EPOCHS

# A mentési gyakoriság beállítása a kérésnek megfelelően
# save_steps = max(1, min(steps_per_epoch // 2, 6))
save_steps = 6

print(f"Dinamikusan számított max_steps: {max_steps} ({NUM_EPOCHS} epoch-hoz)")
print(f"Mentési gyakoriság (save_steps): {save_steps}")

training_arguments = SFTConfig(
    output_dir=output_dir,
    max_steps=max_steps,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    save_steps=save_steps,
    logging_steps=save_steps,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    dataset_text_field="text",
    max_length=1024,
    seed=42,
    save_total_limit=2,
)

# --- Tréner inicializálása ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=PEFT_CONFIG,
    processing_class=tokenizer,
    args=training_arguments,
)

# --- Tanítás indítása ---
print("A finomhangolás elindítása...")
# A resume_from_checkpoint=True argumentum biztosítja, hogy a Trainer
# automatikusan betöltse a legutóbbi checkpointot, ha létezik.
trainer.train(resume_from_checkpoint=should_resume_from_checkpoint(output_dir))
print("A finomhangolás befejeződött.")

# --- Modell mentése ---
print(f"A finomhangolt modell (adapter) mentése a '{NEW_MODEL_NAME}' könyvtárba...")
trainer.model.save_pretrained(NEW_MODEL_NAME)
# trainer.save_model(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)
print("Modell mentve.")

# --- Memória felszabadítása az összefésülés előtt ---
print("Memória felszabadítása az összefésülés előtt...")
del model, base_model
del trainer
torch.cuda.empty_cache()
print("Memória felszabadítva.")

# --- Önállóan betölthető modell mentése ---
print("A LoRA adapter és a bázismodell összefésülése...")

# A bázismodell újratöltése kvantálással, hogy elférjen a memóriában
merged_model_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=QUANTIZATION_CONFIG,
    device_map="auto",
    max_memory=MAX_MEMORY,
    dtype=torch.half,
    low_cpu_mem_usage=False,
    offload_folder="./offload",
    trust_remote_code=True,
)

# A finomhangolt LoRA adapter betöltése
# A `PeftModel` automatikusan kezeli a kvantált bázismodellt
peft_model = PeftModel.from_pretrained(merged_model_base, NEW_MODEL_NAME)

# Az adapter súlyainak összefésülése a bázismodellel
# A `merge_and_unload` metódus a kvantált modellen is működik
merged_model = peft_model.merge_and_unload()
print("Az összefésülés befejeződött.")

# Az összefésült modell mentése
MERGED_MODEL_PATH = f"{NEW_MODEL_NAME}-merged"
print(f"Az összefésült, önállóan betölthető modell mentése a '{MERGED_MODEL_PATH}' könyvtárba...")
merged_model.save_pretrained(MERGED_MODEL_PATH)
tokenizer.save_pretrained(MERGED_MODEL_PATH)

print("Az önálló modell mentése befejeződött.")
