from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

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

with open("../text/1.txt", "r", encoding="utf-8") as file:
    hungarian_text = file.read()

print("load model...")
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B")
print("prepare tokenizer...")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")
print("-----------------------------------------------------")
tokenizer.src_lang = "hu"
encoded_hi = tokenizer(hungarian_text, return_tensors="pt")
generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("en"))
hu_to_en = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(hu_to_en)
print("-----------------------------------------------------")
print("Hungarian to English:", hu_to_en[0])
