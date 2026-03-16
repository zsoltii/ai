import csv
import os
import random
import re
import shutil
import sys
import threading
import time
from queue import Queue

import ollama

SYSTEM_PROMPT = "You are a professional synthetic data generator designed to create high-quality English training datasets for machine translation. Your goal is to produce linguistically complex, domain-specific text that maximizes lexical variety."

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math, torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig
import ctranslate2
from ctranslate2.converters import TransformersConverter

from util.log_available_gpus import log_available_gpus
from util.hfh_login import hfh_login
from util.finetune import (
    should_resume_from_checkpoint,
    create_new_model_name,
    get_last_checkpoint,
    QUANTIZATION_CONFIG,
    RESULTS_DIRECTORY,
    PEFT_CONFIG,
)
from util.text_generator import TOPIC_LIST, STYLE_MODIFIERS

hfh_login()
log_available_gpus()

# Körkörös téma index kezelése randomizált listával
_shuffled_topics = None
_topic_index = 0


def _initialize_shuffled_topics():
    """Inicializálja a randomizált téma listát."""
    global _shuffled_topics
    _shuffled_topics = TOPIC_LIST.copy()
    random.shuffle(_shuffled_topics)


def _get_next_topic():
    """Visszaadja a következő témát a körkörös, randomizált listából."""
    global _topic_index, _shuffled_topics
    if _shuffled_topics is None:
        _initialize_shuffled_topics()

    # Ha végére értünk a listának, újra randomizálunk
    if _topic_index >= len(_shuffled_topics):
        random.shuffle(_shuffled_topics)
        _topic_index = 0

    topic = _shuffled_topics[_topic_index]
    _topic_index += 1
    return topic


def get_role_user():
    """Generál egy ROLE_USER promptot körkörösen a randomizált TOPIC_LIST-ből."""
    topic = _get_next_topic()
    current_style = random.choice(STYLE_MODIFIERS)
    return (
        f"Task: Generate a raw English text block strictly about '{topic}'.\n"
        f"Tone/Style: {current_style}.\n\n"
        "--- Constraints:\n"
        "- Total Length: 20-50 sentences.\n"
        "- Structural Variety: Mix complex, multi-clause sentences with short, imperative, and interrogative ones.\n"
        "- Vocabulary: Use professional jargon and niche terms related to the topic. Avoid generic fillers.\n"
        "- Sentence Limit: No single sentence may exceed 60 words.\n"
        "- Formatting: RAW TEXT ONLY. No numbers, no paragraph titles, no 'Paragraph 1:', no intro/outro.\n"
        "- No Separators: The blocks must flow back-to-back.\n\n"
        "Start immediately with the content."
    )


# --- Modell és Tokenizer beállítások ---
# BASE_MODEL_ID = "meta-llama/Llama-3.1-8B"
# BASE_MODEL_ID = "meta-llama/Llama-3.2-3B"
# BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"
# BASE_MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
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
    "ollama": {
        "api_url": "http://localhost:11434/api/generate",
    },
}

# Az Ollama-ban használt modell nevek listája (véletlenszerűen választunk)
OLLAMA_MODEL_NAMES = [
    "qwen3:8b",
    "qwen3-vl:8b",
    # "qwen3.5:9b",
    # "gpt-oss:20b",
    # "mistral-small3.2:latest", # not so fast :(
    # "glm-4.7-flash:latest", # too big for 3090 if translator inside the GPU :(
    # "ministral-3:14b", # too slow :(
    # "magistral:latest", # not so fast :(
]

STORY_GENERATOR_MODEL_ID = "ollama"

# --- Iterációk száma ---
ITERATION_COUNT = 10

# --- ctranslate2 eszköz beállítás ---
# Lehetséges értékek: "auto" (automatikus felismerés), "cuda", "cpu"
CT2_DEVICE_TYPE = "auto"

GENERATION_BATCH_SIZE = 2
STORY_COUNT = GENERATION_BATCH_SIZE * 50

# --- CTranslate2 fordítási batch méret ---
# A mondatok hányasával fordítja a ctranslate2 (alapértelmezett: 32)
CT2_TRANSLATION_BATCH_SIZE = 10

PER_DEVICE_TRAIN_BATCH_SIZE = 10
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 4

# A finomhangolt modell mentési neve (LoRA adapter)
NEW_MODEL_NAME = create_new_model_name(BASE_MODEL_ID, "finetuned")
# A finomhangolt LoRA adapter könyvtára
ADAPTER_MODEL_PATH = "./" + NEW_MODEL_NAME

# --- ctranslate2 beállítások ---
CT2_MODEL_ID = "facebook/nllb-200-distilled-1.3B"
CT2_MODEL_PATH = "nllb-200-distilled-1.3B-ct2"

# A forrásnyelv kódja (NLLB formátum).
SOURCE_LANGUAGE = "eng_Latn"
# A célnyelv kódja (NLLB formátum).
TARGET_LANGUAGE = "hun_Latn"

MAX_TOKEN_LENGTH = 350

# --- Memória és Offload beállítások ---
# MAX_MEMORY = {
#     0: "23Gib",  # Memória korlát a 0-s GPU-ra
#     "cpu": "20Gib"  # Memória korlát a CPU-ra (offload esetén)
# }
os.makedirs("./offload", exist_ok=True)


# print("max_memory konfiguráció használata:", MAX_MEMORY)


# --- Adathalmaz generálása (in-memory) ---
def generate_translation_dataset(story_count, batch_size):
    data = []
    generated_story_count = 0

    # -- opus betöltése ctranslate2-vel ---
    print(f"'{CT2_MODEL_ID}' modell előkészítése CTranslate2-höz...")

    # Determine device for CTranslate2
    if CT2_DEVICE_TYPE == "auto":
        try:
            cuda_count = ctranslate2.get_cuda_device_count()
        except Exception:
            cuda_count = 0
        device_type = "cuda" if cuda_count > 0 else "cpu"
        print(f"CTranslate2 detected {cuda_count} CUDA devices. Using: {device_type}")
    else:
        device_type = CT2_DEVICE_TYPE
        print(f"CTranslate2 device manually set to: {device_type}")

    # Kvantálás beállítása az eszköz típusa alapján
    if device_type == "cuda":
        # quantization = "int8_bfloat16" # instable
        quantization = "int8"
    else:
        quantization = "int8"
    print(f"CTranslate2 quantization: {quantization}")

    # Check if conversion is needed
    if not os.path.exists(CT2_MODEL_PATH):
        print(f"Converting model {CT2_MODEL_ID} to CTranslate2 format...")
        converter = TransformersConverter(CT2_MODEL_ID)
        converter.convert(
            output_dir=CT2_MODEL_PATH, quantization=quantization, force=True
        )
        print(f"Model converted to {CT2_MODEL_PATH} with quantization {quantization}.")

    print("Loading tokenizer...")
    ct2_tokenizer = AutoTokenizer.from_pretrained(CT2_MODEL_ID, use_fast=True)
    ct2_tokenizer.src_lang = SOURCE_LANGUAGE

    print(f"Loading CTranslate2 translator on {device_type}...")
    try:
        if device_type == "cpu":
            translator = ctranslate2.Translator(
                CT2_MODEL_PATH,
                device=device_type,
                compute_type=quantization,
                intra_threads=os.cpu_count(),
            )
        else:
            translator = ctranslate2.Translator(
                CT2_MODEL_PATH, device=device_type, compute_type=quantization
            )
    except RuntimeError as e:
        if device_type == "cuda":
            print(f"Failed to initialize CTranslate2 with CUDA: {e}")
            print("Falling back to CPU.")
            device_type = "cpu"
            quantization = "int8"
            translator = ctranslate2.Translator(
                CT2_MODEL_PATH,
                device=device_type,
                compute_type=quantization,
                intra_threads=os.cpu_count(),
            )
        else:
            raise e

    print(f"Translator loaded on {device_type}.")

    # --- Ollama API beállítások ---
    print(f"Ollama API használata. Elérhető modellek: {', '.join(OLLAMA_MODEL_NAMES)}")

    def generate_with_ollama(prompt, system_prompt="", model_name=None):
        """Ollama API hívás szöveg generálásához"""
        if model_name is None:
            model_name = random.choice(OLLAMA_MODEL_NAMES)

        try:
            response = ollama.generate(
                model=model_name,
                prompt=prompt,
                system=system_prompt if system_prompt else "",
                options={
                    "temperature": 0.9,
                    "top_p": 0.95,
                    "top_k": 60,
                    "repeat_penalty": 1.15,
                    "num_predict": 8096,  # Maximális tokenek száma a válaszban
                    "num_ctx": 32 * 1024,  # Kontextusablak mérete
                    "stop": [],  # Ne álljon meg előre definiált stop szavaknál
                },
            )
            return response["response"], response.get("eval_count", 0), model_name
        except Exception as e:
            print(f"Ollama API hiba ({model_name}): {e}")
            return "", 0, model_name

    def unload_ollama_model(model_name=None):
        """Ollama modell memóriából való eltávolítása"""
        if model_name is None:
            # Minden modellt eltávolítunk
            for model in OLLAMA_MODEL_NAMES:
                unload_ollama_model(model)
            return

        try:
            ollama.generate(model=model_name, prompt="", keep_alive=0)
            print(f"[Ollama] {model_name} modell memóriából eltávolítva")
        except Exception as e:
            print(f"[Ollama] Hiba a {model_name} eltávolításakor: {e}")

    def check_and_pull_model(model_name):
        """Ellenőrzi, hogy a modell létezik-e, ha nem, letölti"""
        try:
            # Ellenőrizzük, hogy a modell létezik-e
            models = ollama.list()
            model_names = [m.model for m in models.models]

            if model_name in model_names:
                print(f"[Ollama] {model_name} modell már elérhető")
                return True

            # Modell letöltése
            print(f"[Ollama] {model_name} modell letöltése...")
            ollama.pull(model_name)
            print(f"[Ollama] {model_name} modell sikeresen letöltve")
            return True

        except Exception as e:
            print(f"[Ollama] Hiba a {model_name} modell letöltésekor: {e}")
            return False

    # --- CSV kimenet előkészítése ---
    csv_path = os.path.join(
        os.path.dirname(__file__),
        "../text/hu/csv/en-hu-generated-story-sentences-nllb-200.csv",
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    csv_file = open(csv_path, "a", encoding="utf-8", newline="")
    csv_writer = csv.writer(csv_file)
    if not file_exists:
        csv_writer.writerow(["en", "hu"])
    print(f"Generált mondatok mentése ide: {csv_path}")

    # --- Párhuzamos generálás és fordítás ---
    # Queue-k a szálak közötti kommunikációhoz
    generation_queue = Queue()  # Ollama -> Feldolgozó
    translation_queue = Queue()  # Feldolgozó -> Fordító -> Eredmény

    # Eredmények tárolása
    results_lock = threading.Lock()
    all_results = []
    generation_done = threading.Event()
    translation_done = threading.Event()

    def process_story(story_text):
        """Történet feldolgozása mondatokra bekezdésenként"""
        clean_story = re.sub(r" <\|.*?>", "", story_text, flags=re.DOTALL).strip()

        paragraphs_sentences = []  # Lista a bekezdések mondataiból
        source_text_lines = clean_story.split("\n")
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
                para_sentences = []
                raw_sentences = re.split(r"(?<=[.!?])\s+", para)
                for s in raw_sentences:
                    s = s.strip()
                    s = s.replace(" .", ".")

                    if len(s) < 5:
                        continue
                    if len(s.split()) > 60:
                        print(
                            f"Skipping long sentence ({len(s.split())} words): {s[:50]}..."
                        )
                        continue
                    if not re.search(r'[.!?]["\']?$', s):
                        print(f"Skipping incomplete sentence: {s}")
                        continue
                    para_sentences.append(s)
                if para_sentences:
                    paragraphs_sentences.append(para_sentences)
        return paragraphs_sentences, clean_story

    def translate_sentences(sentences, batch_size=CT2_TRANSLATION_BATCH_SIZE):
        """Mondatok fordítása CTranslate2-vel batch-ekben"""
        if not sentences:
            return []

        all_hu_sentences = [""] * len(sentences)
        source_tokens_list = [
            ct2_tokenizer.convert_ids_to_tokens(ct2_tokenizer.encode(text))
            for text in sentences
        ]

        valid_indices = []
        valid_source_tokens = []

        for idx, tokens in enumerate(source_tokens_list):
            if len(tokens) <= MAX_TOKEN_LENGTH:
                valid_indices.append(idx)
                valid_source_tokens.append(tokens)

        if valid_source_tokens:
            # Batch-ekre bontás és fordítás
            total_batches = (len(valid_source_tokens) + batch_size - 1) // batch_size
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(valid_source_tokens))

                batch_tokens = valid_source_tokens[start_idx:end_idx]
                batch_valid_indices = valid_indices[start_idx:end_idx]

                results = translator.translate_batch(
                    batch_tokens,
                    target_prefix=[[TARGET_LANGUAGE]] * len(batch_tokens),
                    beam_size=1,
                )

                decoded_sentences = [
                    ct2_tokenizer.decode(
                        ct2_tokenizer.convert_tokens_to_ids(result.hypotheses[0]),
                        skip_special_tokens=True,
                    )
                    for result in results
                ]

                for idx, sent in zip(batch_valid_indices, decoded_sentences):
                    all_hu_sentences[idx] = sent

        return all_hu_sentences

    def ollama_generation_thread():
        """Ollama generáló szál"""
        # Egyszer választunk modellt a teljes futásra
        selected_model = random.choice(OLLAMA_MODEL_NAMES)
        print(
            f"[Ollama Thread] Generáló szál elindult. Kiválasztott modell: {selected_model}"
        )

        # Modell ellenőrzése és letöltése ha szükséges
        if not check_and_pull_model(selected_model):
            print(
                f"[Ollama Thread] HIBA: Nem sikerült letölteni a {selected_model} modellt!"
            )
            generation_queue.put(None)
            generation_done.set()
            return

        print(f"[Ollama Thread] {selected_model} modell készen áll a generálásra")

        for story_idx in range(story_count):
            start_time = time.time()
            # Minden generáláskor új random témát választunk
            story, token_count, model_name = generate_with_ollama(
                get_role_user(),
                SYSTEM_PROMPT,
                selected_model,
            )
            end_time = time.time()

            tokens_per_second = (
                token_count / (end_time - start_time)
                if (end_time - start_time) > 0
                else 0
            )
            print(
                f"[Ollama Thread] Történet {story_idx + 1}/{story_count} kész - {tokens_per_second:.2f} tokens/sec"
            )

            generation_queue.put((story, token_count, end_time - start_time))

        generation_queue.put(None)  # Jelzés a végéről
        generation_done.set()
        print("[Ollama Thread] Generálás befejezve")

        # Ollama modell eltávolítása a memóriából (azonnal a generálás végén)
        print(
            f"[Ollama Thread] {selected_model} modell memóriából való eltávolítása..."
        )
        unload_ollama_model(selected_model)

        # Várunk, amíg a fordító szál is befejezi
        print("[Ollama Thread] Várakozás a fordítás befejezésére...")
        translation_done.wait()

    def translation_thread():
        """Fordító szál"""
        print("[Translation Thread] Fordító szál elindult")
        total_tokens = 0
        total_time = 0
        stories_processed = 0
        total_sentences_translated = 0
        translation_start_time = time.time()

        while True:
            item = generation_queue.get()
            if item is None:
                generation_queue.put(None)  # Továbbadjuk a következőnek
                break

            story, token_count, gen_time = item
            total_tokens += token_count
            total_time += gen_time
            stories_processed += 1

            # Feldolgozás
            paragraphs_sentences, clean_story = process_story(story)

            print("--------------------")
            print(f" - Original Full English story:\n{story}\n")
            print("--------------------")
            print(f" - Clean Full English story:\n{clean_story}\n")
            print(
                f" - Generation speed: {total_tokens / total_time:.2f} tokens/sec (átlag)"
            )

            if paragraphs_sentences:
                # Összes mondat összegyűjtése fordításhoz (bekezdés struktúra megőrzése)
                all_sentences = []
                for para_sentences in paragraphs_sentences:
                    all_sentences.extend(para_sentences)

                # Fordítás
                hu_sentences = translate_sentences(all_sentences)

                # Visszaépítjük a bekezdés struktúrát
                hu_sentences_iter = iter(hu_sentences)
                full_en_paragraphs = []
                full_hu_paragraphs = []
                all_en_sentences_for_csv = []  # Egyes mondatok a CSV-hez
                all_hu_sentences_for_csv = []  # Egyes mondatok a CSV-hez
                sentences_count = 0

                for para_sentences in paragraphs_sentences:
                    para_en_sentences = []
                    para_hu_sentences = []

                    for en_sent in para_sentences:
                        hu_sent = next(hu_sentences_iter, "")
                        if hu_sent:
                            para_en_sentences.append(en_sent)
                            para_hu_sentences.append(hu_sent)
                            sentences_count += 1

                    if para_en_sentences:
                        full_en_paragraphs.append(" ".join(para_en_sentences))
                        full_hu_paragraphs.append(" ".join(para_hu_sentences))
                        # Tároljuk az egyes mondatokat is a CSV-hez
                        all_en_sentences_for_csv.extend(para_en_sentences)
                        all_hu_sentences_for_csv.extend(para_hu_sentences)

                total_sentences_translated += sentences_count

                if full_en_paragraphs:
                    # Bekezdések összefűzése új sorokkal (eredeti formázás megőrzése)
                    full_en_text = "\n\n".join(full_en_paragraphs)
                    full_hu_text = "\n\n".join(full_hu_paragraphs)

                    # Eredmények mentése
                    with results_lock:
                        # CSV-be mentjük az egyes mondatokat (nem a bekezdéseket)
                        for en_sent, hu_sent in zip(
                            all_en_sentences_for_csv, all_hu_sentences_for_csv
                        ):
                            all_results.append((en_sent, hu_sent))
                            csv_writer.writerow([en_sent, hu_sent])

                        # Dataset-be mentjük a teljes szöveget
                        structured_text = (
                            "instruction: Translate the following English text to Hungarian\n"
                            f"input-english: {full_en_text}\n"
                            f"output-hungarian: {full_hu_text}"
                        )
                        data.append({"text": structured_text})

                        print("--------------------")
                        print(f" - Full English text:\n{full_en_text}\n")
                        print(f" - Full Hungarian text:\n{full_hu_text}\n")

                    csv_file.flush()
                    print(
                        f"[Translation Thread] {sentences_count} mondat fordítva és {len(full_en_paragraphs)} bekezdésben összefűzve"
                    )
                else:
                    print(
                        f"[Translation Thread] Nincs érvényes fordítás ebben a történetben"
                    )

            pbar.update(1)

        translation_done.set()
        total_translation_time = time.time() - translation_start_time
        sentences_per_second = (
            total_sentences_translated / total_translation_time
            if total_translation_time > 0
            else 0
        )
        print("[Translation Thread] Fordítás befejezve")
        print(
            f"[Translation Thread] Összesen {total_sentences_translated} mondat fordítva {total_translation_time:.2f} másodperc alatt ({sentences_per_second:.2f} mondat/sec)"
        )

    # --- Párhuzamos futtatás ---
    num_batches = math.ceil(story_count / batch_size)
    with tqdm(total=story_count, desc="Adathalmaz generálása", unit="történet") as pbar:
        # Szálak indítása
        gen_thread = threading.Thread(target=ollama_generation_thread)
        trans_thread = threading.Thread(target=translation_thread)

        gen_thread.start()
        trans_thread.start()

        # Várakozás a szálakra
        gen_thread.join()
        trans_thread.join()

        generated_story_count = story_count

    csv_file.close()
    del (
        translator,
        ct2_tokenizer,
    )
    torch.cuda.empty_cache()

    return data


# --- Fő iterációs ciklus ---
for iteration in range(ITERATION_COUNT):
    print(f"\n{'=' * 80}")
    print(f"ITERÁCIÓ {iteration + 1}/{ITERATION_COUNT}")
    print(f"{'=' * 80}\n")

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
        # max_memory=MAX_MEMORY,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        offload_folder="./offload",
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    # --- Alapértelmezett Tokenizer betöltése ---
    print("Alapértelmezett Tokenizer betöltése")
    base_tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, trust_remote_code=True
    )
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
        print(
            f"Meglévő adapter és tokanizer betöltése a '{ADAPTER_MODEL_PATH}' könyvtárból a tanítás folytatásához..."
        )
        model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(
            ADAPTER_MODEL_PATH, trust_remote_code=True
        )
        print("Adapter sikeresen betöltve.")

        if os.path.exists(output_dir):
            print(f"Korábbi checkpointok törlése a '{output_dir}' könyvtárból...")
            shutil.rmtree(output_dir)
            print("Checkpointok törölve.")

    elif last_checkpoint:
        checkpoint_path = os.path.join(output_dir, last_checkpoint)
        print(
            f"Meglévő adapter és toknaizer betöltése a '{checkpoint_path}' könyvtárból a tanítás folytatásához..."
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path, trust_remote_code=True
        )
        print("Adapter sikeresen betöltve.")

    else:
        print("Nem található meglévő adapter. Új adapter létrehozása...")
        model = get_peft_model(base_model, PEFT_CONFIG)
        print("Új adapter sikeresen létrehozva.")
        print("A modell felkészítve a PEFT (LoRA) tanításra.")

    # print("\nModell betöltve. A modell elhelyezkedése:")
    # print(model.hf_device_map)

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
        logging_steps=2,  # Sűrűbb logolás, hogy lásd a loss csökkenését
        # --- KRITIKUS MÓDOSÍTÁKOK ---
        learning_rate=5e-5,  # A 2e-4 LoRA-hoz néha sok, a 5e-5 stabilabb fordításhoz
        lr_scheduler_type="cosine",  # A "constant" helyett a "cosine" segít a finomhangolás végén
        warmup_ratio=0.1,  # Magasabb warmup (10%), hogy ne rántsa el a súlyokat az elején
        weight_decay=0.01,  # Erősebb regularizáció a túltanulás ellen
        # ----------------------------
        bf16=True,
        max_grad_norm=1.0,  # 0.3-ról 1.0-ra emelve stabilabb lehet
        dataset_text_field="text",
        max_length=256,  # Fordításnál ritka a 1024 tokenes mondat. A 256 gyorsabb és kevesebb memóriát eszik.
        packing=False,  # Mondatonkénti tanításnál a packing=False javasolt, hogy tiszta határok legyenek
        seed=42,
        save_total_limit=2,
        dataset_kwargs={
            "add_special_tokens": True,  # Fontos a mondatzáró (EOS) tokenek miatt
            "append_concat_token": False,
        },
    )

    # --- Tréner inicializálása ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        # peft_config=PEFT_CONFIG, # rocm esetén kell
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
        # max_memory=MAX_MEMORY,
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
    print(
        f"Az összefésült, önállóan betölthető modell mentése a '{MERGED_MODEL_PATH}' könyvtárba..."
    )
    merged_model.save_pretrained(MERGED_MODEL_PATH)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)

    print("Az önálló modell mentése befejeződött.")
