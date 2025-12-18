import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math, torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig, M2M100ForConditionalGeneration, M2M100Tokenizer
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
SENTENCE_COUNT = 250
SENTENCE_GENERATE_MESSAGES = [
    {"role": "system", "content": "You are a creative linguistic expert. Output ONLY one grammatically correct English sentence. You must vary the sentence types across the entire English spectrum: declarative, interrogative, imperative, exclamatory, conditional (if/then), optative (wishes), or hypothetical. Use diverse structures including passive voice, compound-complex forms, and varied tenses. Do not use any internal reasoning or preamble. Output ONLY the raw text of the sentence between 3 and 100 words."},
    {"role": "user", "content": "Generate one unique, complex, or creative English sentence. Use any sentence type (statement, question, command, exclamation, conditional, or wish). No thinking, no explain, just the sentence."}
]

PER_DEVICE_TRAIN_BATCH_SIZE = 10
GRADIENT_ACCUMULATION_STEPS = 4
# NUM_EPOCHS = 10 # ez az ideális, kb 90%-os pontosság érhető el vele, viszont 13-14 nap a magyar wikipediát feldolgozni egy AMD 6900 XT-vel
NUM_EPOCHS = 10

# --- m2m100 beállítások ---
M2M100_MODEL_ID = "facebook/m2m100_1.2B"
M2M100_QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.half,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.uint8,
    llm_int8_enable_fp32_cpu_offload=True,  # Engedélyezi a CPU-ra történő offload-ot
)
M2M100_SOURCE_LANGUAGE = "en"
M2M100_TARGET_LANGUAGE = "hu"

# --- Memória és Offload beállítások ---
MAX_MEMORY = {
    0: "14Gib",  # Memória korlát a 0-s GPU-ra
    "cpu": "85Gib"  # Memória korlát a CPU-ra (offload esetén)
}
os.makedirs("./offload_m2m100", exist_ok=True)
print("max_memory konfiguráció használata:", MAX_MEMORY)


# --- Adathalmaz generálása (in-memory) ---
def generate_translation_dataset(sentence_count):
    data = []

    # -- m2m100 betöltése ---
    print(f"'{M2M100_MODEL_ID}' modell betöltése 4-bites kvantálással...")
    m2m100_config = AutoConfig.from_pretrained(M2M100_MODEL_ID)

    if hasattr(m2m100_config, "quantization_config"):
        print(f"m2m100 Original quantization_config {m2m100_config.quantization_config}")
        del m2m100_config.quantization_config
        print("m2m100 Original quantization_config deleted!")

    m2m100_model = M2M100ForConditionalGeneration.from_pretrained(
        M2M100_MODEL_ID,
        config=m2m100_config,
        quantization_config=M2M100_QUANTIZATION_CONFIG,
        dtype=torch.half,
        device_map="auto",
        max_memory=MAX_MEMORY,
        offload_folder="./offload_m2m100",
        low_cpu_mem_usage=False,
    )
    m2m100_tokenizer = M2M100Tokenizer.from_pretrained(M2M100_MODEL_ID)
    m2m100_tokenizer.src_lang = M2M100_SOURCE_LANGUAGE

    print("\nm2m100 Modell betöltve. A modell elhelyezkedése:")
    print(m2m100_model.hf_device_map)

    # --- mondat generáló model betöltése ---
    print(f"'{SENTENCE_MODEL_ID}' modell betöltése 4-bites kvantálással...")
    sentence_model_config = AutoConfig.from_pretrained(SENTENCE_MODEL_ID)

    if hasattr(sentence_model_config, "quantization_config"):
        print(f"Original quantization_config {sentence_model_config.quantization_config}")
        del sentence_model_config.quantization_config
        print(f"Original quantization_config deleted!")

    sentence_model_quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.half,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.uint8,
        llm_int8_enable_fp32_cpu_offload=True,  # Engedélyezi a CPU-ra történő offload-ot
    )

    sentence_model = AutoModelForCausalLM.from_pretrained(
        SENTENCE_MODEL_ID,
        config=sentence_model_config,
        quantization_config=sentence_model_quantization_config,
        device_map="auto",  # Engedélyezi az automatikus elosztást a GPU-k és CPU között.
        max_memory=MAX_MEMORY,
        dtype=torch.half,
        low_cpu_mem_usage=False,
        offload_folder="./offload_sentence",
        # Engedélyezi az ideiglenes mentést (offloading) diszkre, ha a CPU RAM elfogy.
    )

    sentence_tokenizer = AutoTokenizer.from_pretrained(SENTENCE_MODEL_ID)

    print(f"\n{SENTENCE_MODEL_ID} Modell betöltve. A modell elhelyezkedése:")
    print(sentence_model.hf_device_map)

    # --- mondatok generálása ---
    for i in range(sentence_count):
        print(f"Adatpont generálása: {i + 1}/{sentence_count}")
        en_sentence_input_text = sentence_tokenizer.apply_chat_template(SENTENCE_GENERATE_MESSAGES, tokenize=False,
                                                                        add_generation_prompt=True)
        en_sentence_inputs = sentence_tokenizer(en_sentence_input_text, return_tensors="pt").to(sentence_model.device)
        print("Angol mondat generálása folyamatban...")
        with torch.no_grad():
            sentence_output = sentence_model.generate(**en_sentence_inputs, max_new_tokens=4096 * 8, do_sample=True,
                                                      temperature=1.2,
                                                      top_p=0.95, top_k=60, repetition_penalty=1.15,
                                                      pad_token_id=sentence_tokenizer.eos_token_id)
        en_sentence = sentence_tokenizer.decode(sentence_output[0], skip_special_tokens=False)
        if "[/INST]" in en_sentence:
            en_sentence = en_sentence.split("[/INST]")[-1].strip()
        if "</s>" in en_sentence:
            en_sentence = en_sentence.replace("</s>", "").strip()
        if "(" in en_sentence:
            en_sentence = en_sentence.partition("(")[0].strip()
        print(f"Angol mondat: {en_sentence}")

        m2m100_inputs = m2m100_tokenizer(en_sentence, return_tensors="pt").to(m2m100_model.device)
        print("Angol mondat fordítása m2m100 modell segítségével folyamatban...")
        with torch.no_grad():
            m2m100_outputs = m2m100_model.generate(**m2m100_inputs,
                                                   forced_bos_token_id=m2m100_tokenizer.get_lang_id(M2M100_TARGET_LANGUAGE),
                                                   max_length=1024)
        hu_sentence = m2m100_tokenizer.batch_decode(m2m100_outputs, skip_special_tokens=True)[
            0]  # this line runs on CPU!
        print(f"Magyar mondat: {hu_sentence}")

        structured_text = (
            "instruction: Translate the following English sentence to Hungarian\n"
            f"input: {en_sentence}\n"
            f"output: {hu_sentence}"
        )

        data.append(
            {"text": structured_text})

    del m2m100_model, m2m100_tokenizer, m2m100_config, sentence_model_config, sentence_model_quantization_config, sentence_model, sentence_tokenizer
    torch.cuda.empty_cache()

    return data


print("Adathalmaz generálása...")
translation_data = generate_translation_dataset(SENTENCE_COUNT)
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
