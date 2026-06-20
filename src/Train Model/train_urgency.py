import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

train_df = pd.read_csv("data/processed/urgency/train.csv")
test_df = pd.read_csv("data/processed/urgency/test.csv")

train_df.head()
train_df.columns

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

train_df = train_df.rename(columns={"urgency_label": "labels"})
test_df = test_df.rename(columns={"urgency_label": "labels"})

train_dataset = Dataset.from_pandas(train_df[["clean_text", "labels"]])
test_dataset = Dataset.from_pandas(test_df[["clean_text", "labels"]])

def tokenize(batch):
    return tokenizer(
        batch["clean_text"],
        padding="max_length",
        truncation=True,
        max_length=128
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

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=8)

training_args = TrainingArguments(
    output_dir="./results_urgency", 
    evaluation_strategy="epoch", 
    save_strategy="epoch", 
    learning_rate=2e-5, 
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8, 
    num_train_epochs=3, 
    weight_decay=0.01,
    logging_dir="./logs_urgency",
    logging_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

model.save_pretrained("models/urgency_classifier")
tokenizer.save_pretrained("models/urgency_classifier")

print("Model and tokenizer saved to models/urgency_classifier")