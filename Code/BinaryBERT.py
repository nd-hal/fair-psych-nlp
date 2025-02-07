import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import evaluate

import numpy as np

from transformers import DataCollatorWithPadding
import torch

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def training_run(task, fold):

    train_df = pd.read_csv(f"./Data/MedianCV/{fold}/{task}_train.txt", sep='\t', names=["label","text"]).dropna()
    train_df["label"] = train_df["label"].replace({-1:0})
    train_ds = Dataset.from_pandas(train_df, split="train")
    eval_df = pd.read_csv(f"./Data/MedianCV/{fold}/{task}_val.txt", sep='\t', names=["label","text"]).dropna()
    eval_df["label"] = eval_df["label"].replace({-1:0})
    eval_ds = Dataset.from_pandas(eval_df, split="train")
    test_df = pd.read_csv(f"./Data/MedianCV/{fold}/{task}_test.txt", sep='\t', names=["label","text"]).dropna()
    test_df["label"] = test_df["label"].replace({-1:0})
    test_ds = Dataset.from_pandas(test_df, split="test")

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    def tokenizeTest(examples):
        return tokenizer(examples["text"], return_tensors="pt", padding="max_length", truncation=True)

    train_ds = train_ds.map(tokenize_function, batched=True)
    eval_ds = eval_ds.map(tokenize_function, batched=True)
    test_ds = test_ds.map(tokenizeTest, batched=True)

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}

    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-uncased", 
        num_labels=2,
        id2label=id2label, 
        label2id=label2id, 
        torch_dtype="auto"
    )

    training_args = TrainingArguments(
        output_dir="test_trainer", 
        eval_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        weight_decay=0.01,
        save_strategy="epoch", 
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    with torch.no_grad():
        logits = model(**test_ds).logits
        predicted_class_id = logits.argmax(dim=1).item()
        for p in predicted_class_id:
            print(model.config.id2label[p])


fold = 1
task = "Anxiety"
training_run(task, fold)

