from datasets import load_dataset

# Load the sms_spam dataset
# use the train_test_split method to split it into train and test
dataset=load_dataset('sms_spam', split='train').train_test_split(test_size=0.25, shuffle=True, seed=23)

splits = ["train", "test"]

print(dataset['train'][0])