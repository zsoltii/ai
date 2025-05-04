from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import re

UE_OE_OE_UE_UE_ = {
    "ö": "oe",
    "Ö": "Oe",
    "ü": "ue",
    "Ü": "Ue",
    "ő": "oe",
    "Ő": "Oe",
    "ű": "ue",
    "Ű": "Ue"
}

# with open("text/csak egy mező.txt", "r", encoding="utf-8") as file:
#     hungarian_text = file.read()
#
# model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B")
# tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")
#
# tokenizer.src_lang = "hu"
# encoded_hi = tokenizer(hungarian_text, return_tensors="pt")
# generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("en"))
# hu_to_en = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
# print("Hungarian to English:", hu_to_en)


def convert_umlauts(text):
    replacements = UE_OE_OE_UE_UE_
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text

print("load model...")
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B")
print("prepare tokenizer...")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")

hu_to_en = ""
with open("../text/1.txt", "r", encoding="utf-8") as file:
    for line_number, hungarian_text in enumerate(file, start=1):
        print("-----------------------------------------------------")
        print(f"Processing line number: {line_number}")
        print("-----------------------------------------------------")
        print(f"Hungarian text: {hungarian_text}")
        print("-----------------------------------------------------")
        if hungarian_text.strip() == "":
            line_en = "\n"
            hu_to_en += line_en
        else:
            #hungarian_text = convert_umlauts(hungarian_text)
            sentences = re.split(r'(?<=[.!?]) +', hungarian_text)
            for sentence in sentences:
                encoded_hi = tokenizer(sentence, return_tensors="pt")
                generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("en"))
                decoded_lines = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                for translated_line in decoded_lines:
                    print(f"English translation: {translated_line}")
                line_en = " ".join(decoded_lines)
        # print(f"English translation: {line_en}")
        # print("-----------------------------------------------------")
        print("-----------------------------------------------------")
        hu_to_en += line_en + "\n"
print("Hungarian to English:", hu_to_en)