from transformers import TrainingArguments

def get_training_args():
    return TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Increased for your 64GB RAM
        per_device_eval_batch_size=2,   # Increased for evaluation
        logging_dir='./logs',
        logging_steps=10,               # Adjusted for CPU training pace
        logging_first_step=True,
        logging_strategy="steps",
        evaluation_strategy="steps",
        eval_steps=100,                 # Reduced frequency for CPU
        save_strategy="steps",
        save_steps=100,                 # Reduced frequency for CPU
        load_best_model_at_end=True,
        disable_tqdm=True,
        report_to="none",
        dataloader_num_workers=8,       # Added to utilize i9's cores
        gradient_accumulation_steps=4    # Added for stability
    )    