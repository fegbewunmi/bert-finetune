# Full fine-tuning of Bert model

- Created a BERT sentiment classifier (actually DistilBERT) using the Hugging Face Transformers library.  
- Used the IMDB movie review dataset to complete a full fine-tuning and evaluated my model.

The IMDB dataset contains movie reviews that are labeled as either positive or negative.

+ Model: BERT
+ Evaluation approach: `evaluate` method with a Hugging Face `Trainer`
+ Fine-tuning dataset: IMDB dataset
