from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
import numpy as np
import pandas as pd


# Load the sms_spam dataset
# use the train_test_split method to split it into train and test
dataset=load_dataset('sms_spam', split='train').train_test_split(test_size=0.2, shuffle=True, seed=23)

splits = ["train", "test"]

print(dataset['train'][0])

#preprocess the data
#convert all the text into tokens for our models

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Let's use a lambda function to tokenize all the examples
tokenized_dataset = {}
for split in splits:
    tokenized_dataset[split] = dataset[split].map(
        lambda x: tokenizer(x["sms"], truncation=True), batched=True
    )

# print(tokenized_dataset["train"])

# Load and set up the model
# In this case we are doing a full fine tuning, so we will want to unfreeze all parameters.
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2, id2label={0:'not spam', 1:'spam'}, label2id={'not spam':0, 'spam':1})

# Unfreeze all the model parameters.
for param in model.base_model.parameters():
    param.requires_grad = True

# print(model)

#train model using Trainer class
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./data/spam_not_spam",
        # Set the learning rate
        learning_rate=2e-5,
        # Set the per device train batch size and eval batch size
        per_device_train_batch_size=16,  
        per_device_eval_batch_size=16, 
        # Evaluate and save the model after each epoch
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
    ),
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()

#Evaluate model
trainer.evaluate()

#View results in a dataframe
# Make a dataframe with the predictions and the text and the labels

items_for_manual_review = tokenized_dataset["test"].select(
    [0, 1, 22, 31, 43, 292, 448, 487]
)

results = trainer.predict(items_for_manual_review)
df = pd.DataFrame(
    {
        "sms": [item["sms"] for item in items_for_manual_review],
        "predictions": results.predictions.argmax(axis=1),
        "labels": results.label_ids,
    }
)
# Show all the cell
pd.set_option("display.max_colwidth", None)
df
print(df)