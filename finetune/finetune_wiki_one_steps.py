import os
import sys

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math, torch
import xml.etree.ElementTree as ET
from datasets import IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig

from util.log_available_gpus import log_available_gpus
from util.hfh_login import hfh_login
from util.finetune import count_wiki_documents, WIKI_NS, HU_WIKI_BASE_URL
from util.finetune import should_resume_from_checkpoint, create_new_model_name, get_last_checkpoint, \
    QUANTIZATION_CONFIG, RESULTS_DIRECTORY, PEFT_CONFIG

hfh_login()
# log_available_gpus()

# --- Modell és Tokenizer beállítások ---
# BASE_MODEL_ID = "meta-llama/Llama-3.1-8B" # TPS: 28s/it; ~4400h
# BASE_MODEL_ID = "meta-llama/Llama-3.2-3B" # TPS: 12-13s/it; ~1950h
# BASE_MODEL_ID = "meta-llama/Llama-3.2-1B" # TPS: 5s/it; ~780h
# BASE_MODEL_ID = "Qwen/Qwen3-1.7B" # TPS: 6-7s/it; ~1100h;
# BASE_MODEL_ID = "Qwen/Qwen3-4B" # TPS: 19-20s/it; ~3000h
# BASE_MODEL_ID = "Qwen/Qwen3-8B" # TPS: 28-29s/it; ~4600h
# BASE_MODEL_ID = "TinyLlama/TinyLlama_v1.1" # TPS: 5-6s/it; ~850h
# BASE_MODEL_ID = "HuggingFaceTB/SmolLM3-3B" # TPS: 12s/it; ~1900s
BASE_MODEL_ID = "google/gemma-3-1b-it" # TPS: 2s/it; 650h
# BASE_MODEL_ID = "google/gemma-3-4b-it" # TPS: 16-17s/it; ~2600h
# BASE_MODEL_ID = "microsoft/Phi-4-mini-reasoning" # TPS: 15-16s/it; ~2500h
# BASE_MODEL_ID = "ibm-granite/granite-4.0-micro" # TPS: 15-16s/it; ~2500h
# BASE_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" # TPS: 6-7s/it; ~1000h
# A finomhangolt modell mentési neve (LoRA adapter)
NEW_MODEL_NAME = create_new_model_name(BASE_MODEL_ID, "finetuned")
# A finomhangolt LoRA adapter könyvtára
ADAPTER_MODEL_PATH = "./" + NEW_MODEL_NAME
# A lokális adathalmazt tartalmazó főkönyvtár
DATASET_PATH = "../wikiextractor/hu/huwiki-latest-pages-articles.xml"

PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2
NUM_EPOCHS = 4

# --- Memória és Offload beállítások ---
MAX_MEMORY = {
    0: "14Gib",  # Memória korlát a 0-s GPU-ra
    "cpu": "85Gib"  # Memória korlát a CPU-ra (offload esetén)
}
os.makedirs("./offload", exist_ok=True)
print("max_memory konfiguráció használata:", MAX_MEMORY)

# --- Modell betöltése ---
print(f"'{BASE_MODEL_ID}' modell betöltése...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=QUANTIZATION_CONFIG,
    device_map="auto",
    max_memory=MAX_MEMORY,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
    offload_folder="./offload",
    trust_remote_code=True,
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# --- Alapértelmezett Tokenizer betöltése ---
print("Alapértelmezett Tokenizer betöltése")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- Adatfolyam (Streaming) beállítása ---
def stream_data_from_local_files(xmlPath):
    for event, elem in ET.iterparse(xmlPath, events=('end',)):
        if elem.tag.endswith('page'):
            # Ellenőrizzük a névteret (0 = szócikk)
            ns = elem.find(f'{WIKI_NS}ns')
            is_redirect = elem.find(f'{WIKI_NS}redirect') is not None

            if ns is not None and ns.text == '0':
                if not is_redirect:
                    revision = elem.find(f'{WIKI_NS}revision')
                    if revision is not None:
                        text = revision.find(f'{WIKI_NS}text')
                        if text is not None and text.text != '':
                            title = elem.find(f"{WIKI_NS}title")
                            id = elem.find(f"{WIKI_NS}id")
                            url = f"{HU_WIKI_BASE_URL}{title}"
                            structured_text = (
                                f"Title: {title}\n"
                                f"ID: {id}\n"
                                f"URL: {url}\n\n"
                                f"{text.text}"
                            )
                            yield {"text": structured_text}

# --- Teljes adathalmaz finomhangolása ---
output_dir = RESULTS_DIRECTORY + "-" + NEW_MODEL_NAME
print(f"--- Teljes adathalmaz feldolgozása: {DATASET_PATH} ---")
print(f"--- Eredmények mentése ide: {output_dir} ---")

model = base_model

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

print("Dokumentumok számának meghatározása...")
num_documents = count_wiki_documents(DATASET_PATH)
if num_documents == 0:
    print("Figyelmeztetés: Nem található feldolgozható adat a megadott könyvtárban. A tanítás nem indul el.")
    raise ValueError("Nem található feldolgozható adat a megadott könyvtárban.")
else:
    print(f"Talált dokumentumok száma: {num_documents}")

    dataset = IterableDataset.from_generator(stream_data_from_local_files, gen_kwargs={"xmlPath": DATASET_PATH})
    print("Streaming adathalmaz beállítva.")

    # --- Tanítási Argumentumok (Dinamikus lépésszámmal) ---
    effective_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    steps_per_epoch = math.ceil(num_documents / effective_batch_size)
    max_steps = steps_per_epoch * NUM_EPOCHS

    # A mentési gyakoriság beállítása a kérésnek megfelelően
    save_steps = max(1, min(steps_per_epoch // 2, 100))

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
        logging_steps=6,

        learning_rate=2e-4,  # Megtartjuk a 2e-4-et, de a cosine lecsökkenti majd
        lr_scheduler_type="cosine",  # "constant" helyett "cosine" a lágyabb lecsengésért
        warmup_ratio=0.1,  # 0.03 helyett 0.1 (a tanítás első 10%-a bemelegítés)
        weight_decay=0.01,  # 0.001-ről 0.01-re emelve az overfitting ellen (Wiki esetén fontos)

        bf16=True,
        fp16=False,
        max_grad_norm=0.3,
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
        device_map="auto",
        max_memory=MAX_MEMORY,
        dtype=torch.bfloat16,
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
