import numpy as np
from datasets import load_metric

def get_basic_example_fn(tokenizer, src_cols=[], tar_col='label', label_fn=None, max_src_len=256, truncation=True, padding="max_length"):

    def example_fn(examples):
        output = tokenizer(*[examples[col] for col in [c for c in src_cols if c in examples]],
                           padding=padding,
                           max_length=max_src_len,
                           truncation=True)
        if tar_col in examples:
            output["labels"] = [label_fn(c) for c in examples[tar_col]] if label_fn else examples[tar_col]
        return output
    
    return example_fn

metric_fn = load_metric('glue', 'mnli')

def metric(p):
    preds, labels = p
    if not isinstance(preds, tuple) and not isinstance(preds, list):
        if len(preds.shape) == 2 and preds.shape[1] == 1:
            preds = preds[:, 0]
        elif len(preds.shape) - len(labels.shape) == 1:
            preds = np.argmax(preds, axis=-1)
    return metric_fn.compute(predictions=preds, references=labels)

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding, 
    EarlyStoppingCallback
)
path = 'D:\\Study\\_data\\dacon\\natural\\'
TRAIN = path +"train_data.csv"
TEST = path +"test_data.csv"
CKPT = "runs/"
MODEL = 'klue/roberta-base'

MAX_LEN = 256
TRAIN_BATCH = 32
EVAL_BATCH = 16

# Prepare the train/eval/test dataset.
map_dict = {'contradiction': 0, 'neutral': 1, 'entailment': 2}

tokenizer = AutoTokenizer.from_pretrained(MODEL)

example_fn = get_basic_example_fn(
    tokenizer,
    src_cols=['premise', 'hypothesis'], 
    tar_col='label', 
    label_fn=lambda x: map_dict.get(x),
    max_src_len=MAX_LEN,
    # max_tar_len=MAX_LEN,
    truncation=True, 
    padding="max_length"
    )

dataset = load_dataset("csv", data_files={"train":TRAIN})
dataset = dataset['train'].map(example_fn, remove_columns=['index', 'premise', 'hypothesis', 'label'], batched=True)
dataset = dataset.train_test_split(0.1)
train_data, eval_data = dataset['train'], dataset['test']

# Define the model, arguments, and Trainer with huggingface transformers.
training_arguments = TrainingArguments(
    output_dir="runs/roberta_basic/",
    num_train_epochs=20,
    per_device_train_batch_size=TRAIN_BATCH,
    per_device_eval_batch_size=EVAL_BATCH,
    logging_strategy='epoch',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    save_steps=1000,
    save_total_limit=3,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    load_best_model_at_end=True,
    label_smoothing_factor=0.0,
)

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(map_dict))

trainer = Trainer(
    model=model,
    args=training_arguments,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=metric,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

# Training.
trainer.train()

# Prepare the test data.
test_data = load_dataset("csv", data_files={"test":TEST})
test_data = test_data['test'].remove_columns("label").map(example_fn, batched=True, remove_columns = ['premise', 'hypothesis'])

# Predict the test outputs.
outputs = trainer.predict(test_data)

# Transform the test outputs for the submission.
df_sub = pd.DataFrame({"index":test_data['index'], "label":np.argmax(outputs.predictions, axis=-1)})
df_sub['label'] = df_sub['label'].apply(lambda x: {v:k for k,v in map_dict.items()}[x])
df_sub.to_csv("0209_01.csv", index=False)





















