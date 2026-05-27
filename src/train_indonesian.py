import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

train_df = pd.read_csv("data/processed/indonesian/translated_train.csv")
test_df = pd.read_csv("data/processed/indonesian/translated_test.csv")

train_df.head()
train_df.columns

tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

train_df = train_df.rename(columns={"target": "labels"})
test_df = test_df.rename(columns={"target": "labels"})

train_dataset = Dataset.from_pandas(train_df[["translated_text", "labels"]])
test_dataset = Dataset.from_pandas(test_df[["translated_text", "labels"]])

def tokenize(batch):
    return tokenizer(
        batch["translated_text"],
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

model = AutoModelForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p1", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results_indobert", 
    evaluation_strategy="epoch", 
    save_strategy="epoch", 
    learning_rate=2e-5, 
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8, 
    num_train_epochs=3, 
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

model.save_pretrained("models/indobert")
tokenizer.save_pretrained("models/indobert")

print("Model and tokenizer saved to models/indobert")