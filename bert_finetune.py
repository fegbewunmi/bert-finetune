from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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