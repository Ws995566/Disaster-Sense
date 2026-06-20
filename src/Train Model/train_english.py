import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

train_df = pd.read_csv("data/processed/English/train.csv")
test_df = pd.read_csv("data/processed/English/test.csv")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

train_df = train_df.rename(columns={"target": "labels"})
test_df = test_df.rename(columns={"target": "labels"})

train_dataset = Dataset.from_pandas(train_df[["clean_text", "labels"]])
test_dataset = Dataset.from_pandas(test_df[["clean_text", "labels"]])

def tokenize(batch):
    return tokenizer(
        batch["clean_text"],
        padding="max_length",
        truncation=True
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels"
    ]
)

test_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels"
    ]
)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results", 
    evaluation_strategy="epoch", 
    save_strategy="epoch", 
    learning_rate=2e-5, 
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8, 
    num_train_epochs=3, 
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

model.save_pretrained("models/english_bert")
tokenizer.save_pretrained("models/english_bert")