import os
import sys
import re
import shutil

ROLE_USER = "Write me a random English short story. The sentences long must be maximum 60 words!"
ROLE_SYSTEM = ""

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

# hfh_login()
# log_available_gpus()

# --- Modell és Tokenizer beállítások ---
# --- Modell és Tokenizer beállítások ---
# BASE_MODEL_ID = "meta-llama/Llama-3.1-8B"
# BASE_MODEL_ID = "meta-llama/Llama-3.2-3B"
# BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"
BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
# BASE_MODEL_ID = "Qwen/Qwen3-4B"
# BASE_MODEL_ID = "Qwen/Qwen3-8B"
# BASE_MODEL_ID = "TinyLlama/TinyLlama_v1.1"
# BASE_MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
# BASE_MODEL_ID = "google/gemma-3-1b-it"
# BASE_MODEL_ID = "google/gemma-3-4b-it"
# BASE_MODEL_ID = "microsoft/Phi-4-mini-reasoning"
# BASE_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

STORY_GENERATOR_MODELS = {
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "apply_chat_template": True,
        "role_user": ROLE_USER,
        "role_system": ROLE_SYSTEM,
        "dtype": "nf4"
    },
    "Qwen/Qwen3-1.7B": {
        "apply_chat_template": True,
        "role_user": ROLE_USER,
        "role_system": ROLE_SYSTEM,
        "dtype": "nf4"
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "apply_chat_template": True,
        "role_user": ROLE_USER,
        "role_system": ROLE_SYSTEM,
        "dtype": "bfloat16"
    },
    "meta-llama/Llama-3.2-1B-Instruct": {
        "apply_chat_template": True,
        "role_user": ROLE_USER,
        "role_system": ROLE_SYSTEM,
        "dtype": "bfloat16"
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "apply_chat_template": True,
        "role_user": ROLE_USER,
        "role_system": ROLE_SYSTEM,
        "dtype": "nf4"
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "apply_chat_template": True,
        "role_user": ROLE_USER,
        "role_system": ROLE_SYSTEM,
        "dtype": "nf4"
    },
    "google/gemma-3-1b-it": {
        "apply_chat_template": False,
        "role_user": ROLE_USER,
        # "role_user": ROLE_SYSTEM + "; " + ROLE_USER,
        "role_system": "",
        "dtype": "bfloat16"
    },
    "TinyLlama/TinyLlama_v1.1": {
        "apply_chat_template": False,
        "role_user": ROLE_USER,
        # "role_user": ROLE_SYSTEM + "; " + ROLE_USER,
        "role_system": "",
        "dtype": "bfloat16"
    }
}

STORY_GENERATOR_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
GENERATION_BATCH_SIZE = 2
STORY_COUNT = GENERATION_BATCH_SIZE * 2

PER_DEVICE_TRAIN_BATCH_SIZE = 10
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 4

# A finomhangolt modell mentési neve (LoRA adapter)
NEW_MODEL_NAME = create_new_model_name(BASE_MODEL_ID, "finetuned")
# A finomhangolt LoRA adapter könyvtára
ADAPTER_MODEL_PATH = "./" + NEW_MODEL_NAME

# --- Opus-MT beállítások ---
# OPUS_MODEL_ID = "Helsinki-NLP/opus-mt-en-hu" # ez van használva  sebesség méréshez
OPUS_MODEL_ID = "Helsinki-NLP/opus-mt-tc-big-en-hu" # kicsit lasabb, de sokkal jobb a magyarja

# --- Memória és Offload beállítások ---
MAX_MEMORY = {
    0: "14Gib",  # Memória korlát a 0-s GPU-ra
    "cpu": "85Gib"  # Memória korlát a CPU-ra (offload esetén)
}
os.makedirs("./offload", exist_ok=True)
print("max_memory konfiguráció használata:", MAX_MEMORY)


# --- Adathalmaz generálása (in-memory) ---
def generate_translation_dataset(story_count, batch_size):
    data = []
    generated_story_count = 0

    # -- opus betöltése ---
    print(f"'{OPUS_MODEL_ID}' modell betöltése...")
    opus_model = AutoModelForSeq2SeqLM.from_pretrained(
        OPUS_MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    print("Compiling opus_model with torch.compile()... (first run will be slower)")
    opus_model = torch.compile(opus_model, mode="reduce-overhead")
    opus_tokenizer = AutoTokenizer.from_pretrained(OPUS_MODEL_ID, use_fast=True)
    print("\nOpus-MT Modell betöltve. A modell elhelyezkedése:")
    print(opus_model.hf_device_map)

    # --- mondat generáló model betöltése ---
    print(f"'{STORY_GENERATOR_MODEL_ID}' modell betöltése...")
    sentence_model_config = AutoConfig.from_pretrained(STORY_GENERATOR_MODEL_ID)

    if hasattr(sentence_model_config, "quantization_config"):
        print(f"Original quantization_config {sentence_model_config.quantization_config}")
        del sentence_model_config.quantization_config
        print(f"Original quantization_config deleted!")

    model_config_entry = STORY_GENERATOR_MODELS[STORY_GENERATOR_MODEL_ID]
    model_dtype_str = model_config_entry["dtype"]

    quantization_config = None
    torch_dtype = torch.bfloat16

    if model_dtype_str == "nf4":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.half,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.uint8,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        torch_dtype = torch.half
    elif model_dtype_str == "half":
        torch_dtype = torch.float16
    elif model_dtype_str == "float32":
        torch_dtype = torch.float32
    elif model_dtype_str == "bfloat16":
        torch_dtype = torch.bfloat16

    story_generator_model = AutoModelForCausalLM.from_pretrained(
        STORY_GENERATOR_MODEL_ID,
        dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map="auto",
        attn_implementation="sdpa"
    )
    story_generator_model = torch.compile(story_generator_model, mode="reduce-overhead")
    sentence_tokenizer = AutoTokenizer.from_pretrained(
        STORY_GENERATOR_MODEL_ID,
        use_fast=True,
    )
    # A CausalLM modellekhez a padding token beállítása elengedhetetlen a batch generáláshoz
    if sentence_tokenizer.pad_token is None:
        sentence_tokenizer.pad_token = sentence_tokenizer.eos_token
    sentence_tokenizer.padding_side = "left"  # Left padding for decoder-only models

    print(f"\n{STORY_GENERATOR_MODEL_ID} Modell betöltve. A modell elhelyezkedése:")
    print(story_generator_model.hf_device_map)

    # --- mondatok generálása batch-ekben ---
    num_batches = math.ceil(story_count / batch_size)
    with tqdm(total=story_count, desc="Adathalmaz generálása", unit="történet") as pbar:
        for i in range(num_batches):
            current_batch_size = min(batch_size, story_count - generated_story_count)
            if current_batch_size <= 0:
                break

            pbar.set_description(f"Batch {i + 1}/{num_batches}: Angol történetek generálása")

            if model_config_entry["apply_chat_template"]:
                messages = []
                if model_config_entry["role_system"]:
                    messages.append({"role": "system", "content": model_config_entry["role_system"]})
                messages.append({"role": "user", "content": model_config_entry["role_user"]})

                batch_input_texts = [
                    sentence_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    for _ in range(current_batch_size)
                ]
            else:
                batch_input_texts = [model_config_entry["role_user"]] * current_batch_size

            en_sentence_inputs = sentence_tokenizer(
                batch_input_texts,
                padding="longest",
                pad_to_multiple_of=8,
                return_tensors="pt",
                truncation=True,
            ).to(story_generator_model.device)

            with torch.no_grad():
                sentence_outputs = story_generator_model.generate(
                    **en_sentence_inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=1.2,
                    top_p=0.95,
                    top_k=60,
                    repetition_penalty=1.15,
                    pad_token_id=sentence_tokenizer.eos_token_id,
                    num_return_sequences=1
                )

            generated_ids = sentence_outputs[:, en_sentence_inputs.input_ids.shape[1]:]
            decoded_stories = sentence_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            all_en_sentences = []
            for story in decoded_stories:
                clean_story = re.sub(r'<think>.*?</think>', '', story, flags=re.DOTALL).strip()
                if "(" in clean_story:
                    clean_story = clean_story.partition("(")[0].strip()
                
                print("--------------------")
                print(f" - Full English story:\n{clean_story}")
                
                # Történetek szétbontása mondatokra (m2m100 algoritmus alapján)
                source_text_lines = clean_story.split('\n')
                paragraphs = []
                current_paragraph = ""
                for line in source_text_lines:
                    if not line.strip():
                        if current_paragraph:
                            paragraphs.append(current_paragraph)
                            current_paragraph = ""
                    else:
                        if current_paragraph:
                            current_paragraph += " " + line.strip()
                        else:
                            current_paragraph = line.strip()
                if current_paragraph:
                    paragraphs.append(current_paragraph)

                for para in paragraphs:
                    if para.strip():
                        sentences = re.split(r'(?<=[.!?])\s+', para)
                        for s in sentences:
                            s = s.strip()
                            s = s.replace(" .", ".") # Fix spacing before dot
                            
                            # 1. Szűrő: Túl rövid mondatok (zaj)
                            if len(s) < 5:
                                continue
                                
                            # 2. Szűrő: Túl hosszú mondatok (run-on sentences)
                            # Kb. 60 szó felett már valószínűleg rossz minőségű a generálás vagy túl komplex
                            if len(s.split()) > 60:
                                print(f"Skipping long sentence ({len(s.split())} words): {s[:50]}...")
                                continue
                                
                            # 3. Szűrő: Érvényes mondatvégi írásjel ellenőrzése
                            # Ha nem pontra, felkiáltójelre vagy kérdőjelre végződik (vagy idézőjelre ezek után), akkor kuka.
                            # Ez kiszűri a félbehagyott mondatokat a generálás végén.
                            if not re.search(r'[.!?]["\']?$', s):
                                print(f"Skipping incomplete sentence: {s}")
                                continue

                            all_en_sentences.append(s)

            if not all_en_sentences:
                generated_story_count += len(decoded_stories)
                pbar.update(len(decoded_stories))
                continue

            pbar.set_description(f"Batch {i + 1}/{num_batches}: Fordítás magyarra ({len(decoded_stories)} történet)")
            
            # Fordítás kisebb batch-ekben
            translation_batch_size = 32
            all_hu_sentences = []
            
            opus_tokenizer.padding_side = "right"
            
            for j in range(0, len(all_en_sentences), translation_batch_size):
                batch_slice = all_en_sentences[j : j + translation_batch_size]
                
                opus_inputs = opus_tokenizer(
                    batch_slice,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    max_length=512,
                ).to(opus_model.device)

                with torch.inference_mode():
                    opus_outputs = opus_model.generate(
                        **opus_inputs,
                        num_beams=1,  # Gyorsabb, mint a beam search
                        do_sample=False,
                        use_cache=True,
                    )

                hu_batch = opus_tokenizer.batch_decode(opus_outputs, skip_special_tokens=True)
                all_hu_sentences.extend(hu_batch)

            for en_sentence, hu_sentence in zip(all_en_sentences, all_hu_sentences):
                structured_text = (
                    "instruction: Translate the following English sentence to Hungarian\n"
                    f"input-english: {en_sentence}\n"
                    f"output-hungarian: {hu_sentence}"
                )
                print("--------------------")
                print(f" - English sentence: {en_sentence}")
                print(f" - Hungarian sentence: {hu_sentence}")
                data.append({"text": structured_text})
            
            generated_story_count += len(decoded_stories)
            pbar.update(len(decoded_stories))

    del opus_model, opus_tokenizer, story_generator_model, sentence_tokenizer, sentence_model_config
    torch.cuda.empty_cache()

    return data


print("Adathalmaz generálása...")
translation_data = generate_translation_dataset(STORY_COUNT, GENERATION_BATCH_SIZE)
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

if is_adapter_saved:
    print(f"Meglévő adapter és tokanizer betöltése a '{ADAPTER_MODEL_PATH}' könyvtárból a tanítás folytatásához...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL_PATH, trust_remote_code=True)
    print("Adapter sikeresen betöltve.")
    
    if os.path.exists(output_dir):
        print(f"Korábbi checkpointok törlése a '{output_dir}' könyvtárból...")
        shutil.rmtree(output_dir)
        print("Checkpointok törölve.")

elif last_checkpoint:
    checkpoint_path = os.path.join(output_dir, last_checkpoint)
    print(f"Meglévő adapter és toknaizer betöltése a '{checkpoint_path}' könyvtárból a tanítás folytatásához...")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
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
# save_steps = max(1, min(steps_per_epoch // 2, 100))
save_steps = 6

print(f"Dinamikusan számított max_steps: {max_steps} ({NUM_EPOCHS} epoch-hoz)")
print(f"Mentési gyakoriság (save_steps): {save_steps}")

training_arguments = SFTConfig(
    output_dir=output_dir,
    # 10k mondatnál érdemesebb epoch-alapú tanítást nézni, vagy több ezer lépést
    max_steps=max_steps,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    save_steps=save_steps,
    logging_steps=2, # Sűrűbb logolás, hogy lásd a loss csökkenését

    # --- KRITIKUS MÓDOSÍTÁSOK ---
    learning_rate=5e-5,        # A 2e-4 LoRA-hoz néha sok, a 5e-5 stabilabb fordításhoz
    lr_scheduler_type="cosine", # A "constant" helyett a "cosine" segít a finomhangolás végén
    warmup_ratio=0.1,          # Magasabb warmup (10%), hogy ne rántsa el a súlyokat az elején
    weight_decay=0.01,         # Erősebb regularizáció a túltanulás ellen
    # ----------------------------

    bf16=True,
    max_grad_norm=1.0,         # 0.3-ról 1.0-ra emelve stabilabb lehet
    dataset_text_field="text",
    max_length=256,        # Fordításnál ritka a 1024 tokenes mondat. A 256 gyorsabb és kevesebb memóriát eszik.
    packing=False,             # Mondatonkénti tanításnál a packing=False javasolt, hogy tiszta határok legyenek
    seed=42,
    save_total_limit=2,
    dataset_kwargs={
        "add_special_tokens": True, # Fontos a mondatzáró (EOS) tokenek miatt
        "append_concat_token": False,
    }
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
