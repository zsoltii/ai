import os
import torch
import xml.etree.ElementTree as ET
from transformers import BitsAndBytesConfig
from peft import LoraConfig

WIKI_NS = '{http://www.mediawiki.org/xml/export-0.11/}'
HU_WIKI_BASE_URL = 'https://hu.wikipedia.org/wiki/'

RESULTS_DIRECTORY = "./results"
# --- Kvantálási Konfiguráció (memóriahatékonyságért) ---
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
# --- PEFT (LoRA) Konfiguráció ---
PEFT_CONFIG = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj"
    ]
)

def should_resume_from_checkpoint(results_dir):
    if not os.path.isdir(results_dir):
        return False
    for item in os.listdir(results_dir):
        if os.path.isdir(os.path.join(results_dir, item)) and item.startswith("checkpoint-"):
            print(f"Checkpoint található: {item}. A tanítás folytatódik.")
            return True
    print("Nem található checkpoint. A tanítás a nulláról indul.")
    return False

def create_new_model_name(base_model_id, postfix):
    return base_model_id.replace("/", "-") + "-" + postfix

def get_last_checkpoint(directory):
    """
    Visszaadja a legutóbbi checkpoint könyvtár nevét a megadott könyvtárból.
    """
    if not os.path.exists(directory):
        return None
    checkpoints = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d)) and d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda x: int(x.split('-')[-1]))

# --- Wiki dokumentumok számának meghatározása ---
def count_wiki_documents(xmlPath):
    count = 0
    skip_count = 0

    for event, elem in ET.iterparse(xmlPath, events=('end',)):
        if elem.tag.endswith('page'):
            # Ellenőrizzük a névteret (0 = szócikk)
            ns = elem.find(f'{WIKI_NS}ns')
            is_redirect = elem.find(f'{WIKI_NS}redirect') is not None

            if ns is not None and ns.text == '0':
                if is_redirect:
                    skip_count += 1
                else:
                    revision = elem.find(f'{WIKI_NS}revision')
                    if revision is not None:
                        text = revision.find(f'{WIKI_NS}text')
                        if text is not None and text.text != '':
                            count += 1

            # Memória felszabadítása
            elem.clear()

    print(f"Kihagyott wiki lapok száma: {skip_count}")
    print(f"Talált wiki lapok száma: {count}")
    return count