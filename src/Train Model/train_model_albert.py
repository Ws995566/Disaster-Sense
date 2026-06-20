import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Use the same English processed dataset structure used by other training scripts
train_df = pd.read_csv("data/processed/English/train.csv")
test_df = pd.read_csv("data/processed/English/test.csv")

tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

train_df = train_df.rename(columns={"target": "labels"})
test_df = test_df.rename(columns={"target": "labels"})

train_dataset = Dataset.from_pandas(train_df[["clean_text", "labels"]])
test_dataset = Dataset.from_pandas(test_df[["clean_text", "labels"]])


def tokenize(batch):
    return tokenizer(
        batch["clean_text"],
        padding="max_length",
        truncation=True,
    )


train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels",
    ],
)

test_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels",
    ],
)


model = AutoModelForSequenceClassification.from_pretrained("albert-base-v2", num_labels=2)

# Explicitly enable GPU usage when available (fp16 helps performance on CUDA)
use_cuda = torch.cuda.is_available()
print(f"[train_model_albert] CUDA available: {use_cuda}")
if use_cuda:
    print(f"[train_model_albert] Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[train_model_albert] Using CPU")

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=use_cuda,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Intentionally not running here; keep consistent with other files that call trainer.train().
# To train, run this file and ensure you are using a CUDA-enabled environment.
trainer.train()

model.save_pretrained("models/albert-base-v2")
tokenizer.save_pretrained("models/albert-base-v2")

