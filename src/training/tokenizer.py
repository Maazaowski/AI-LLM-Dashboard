from transformers import GPT2Tokenizer
from typing import Dict, Any
import logging
import datasets
import transformers

# Set logging levels
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

class ModelTokenizer:
    def __init__(self, progress_callback=None, log_callback=None):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.progress_callback = progress_callback
        self.log_callback = log_callback

    def update_terminal_progress(self, current, total):
        if self.progress_callback:
            self.progress_callback(current, total)

    def update_log(self, category, message):
        if self.log_callback:
            self.log_callback(category, message)
            
    def tokenize_dataset(self, dataset, max_length: int = 512):
        total_items = len(dataset)
        
        def tokenize_function(examples):
            current_index = examples['text'].index
            self.update_terminal_progress(current_index, total_items)
            self.update_log("Tokenization", f"Processing items {current_index} of {total_items}")
            
            # Tokenize the inputs
            result = self.tokenizer(
                examples['text'],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Set labels equal to input_ids for language modeling
            result["labels"] = result["input_ids"].detach().clone()
            
            return result
        
        return dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=['text']  # Remove only the 'text' column which exists in the dataset
        )
