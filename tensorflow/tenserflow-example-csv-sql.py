import os

SEP = "###"

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM, create_optimizer

print(tf.executing_eagerly())

# 1. Adatok betöltése és előkészítése
df = pd.read_csv("../csv/sql-mysql-db.csv")

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

@tf.function
def preprocess_function(examples):
    print(tf.executing_eagerly())
    question_ = examples["question"]
    schema_ = examples["schema"]
    print(type(question_))
    inputs = tf.strings.join(["schema: ", schema_, SEP, "question: ", question_, SEP, "answer:"])
    inputs = tf.strings.split(inputs, sep=SEP).numpy().tolist()
    model_inputs = tokenizer(inputs, text_target=examples["chosen"], truncation=True)
    return model_inputs

dataset = tf.data.Dataset.from_tensor_slices(dict(df))
# Assuming `dataset` is already defined
dataset = dataset.map(preprocess_function, num_parallel_calls=tf.data.AUTOTUNE)

# 2. Modell betöltése és finomhangolás beállítása
model = TFAutoModelForSeq2SeqLM.from_pretrained("deepseek-ai/deepseek-coder-1.5b-instruct")

batch_size = 8
num_epochs = 3
total_steps = len(dataset) // batch_size * num_epochs

optimizer, schedule = create_optimizer(
    init_lr=5e-5, num_warmup_steps=0, num_train_steps=total_steps, weight_decay_rate=0.01
)

print(tf.executing_eagerly())
model.compile(optimizer=optimizer, run_eagerly=True)
print(tf.executing_eagerly())

# 3. Finomhangolás
dataset = dataset.shuffle(len(dataset)).batch(batch_size)
model.fit(dataset, epochs=num_epochs)
print(tf.executing_eagerly())

# 4. Modell mentése
model.save_pretrained("./deepseek-r1-fine-tuned")
tokenizer.save_pretrained("./deepseek-r1-fine-tuned")