from transformers import GPT2LMHeadModel, TrainingArguments, Trainer, TrainerCallback
from ..utils.logger import TrainingLogger
from .tokenizer import ModelTokenizer
from datasets import load_dataset
import time

class ModelTrainer:
    def __init__(self, window=None):
        self.window = window
        self.logger = TrainingLogger(window)
        self.tokenizer = ModelTokenizer()
        self.model = None
        self.dataset = None
        self.current_progress = 0

    def update_progress(self, increment: float):
        self.current_progress += increment
        self.logger.update_progress(min(self.current_progress, 100))

    def load_dataset(self):
        self.logger.log("INFO", "Loading and preprocessing dataset...")
        self.dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
        self.logger.log("INFO", f"Dataset loaded. {len(self.dataset['train'])} training examples.")

    def initialize_model(self):
        self.logger.log("INFO", "Initializing model...")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.logger.log("INFO", "Model initialized.")

    def train(self, training_args: TrainingArguments):
        try:
            # Loading dataset (10% of progress)
            self.load_dataset()
            self.update_progress(10)

            # Tokenizing (20% of progress)
            self.logger.log("INFO", "Tokenizing dataset...")
            tokenized_datasets = self.tokenizer.tokenize_dataset(self.dataset)
            self.logger.log("INFO", "Dataset tokenized.")
            self.update_progress(20)

            # Model initialization (10% of progress)
            self.initialize_model()
            self.update_progress(10)

            # Training loop (60% of progress)
            self.logger.log("INFO", "Starting training...")
            num_epochs = training_args.num_train_epochs
            progress_per_epoch = 60 / num_epochs

            class ProgressCallback(TrainerCallback):
                def __init__(self, trainer_instance):
                    self.trainer = trainer_instance
                    self.current_epoch = 0
                    self.progress_per_epoch = progress_per_epoch
                    self.loss_history = []
                    self.accuracy_history = []
                    self.window_size = 10  # Number of recent values to average
        
                def on_epoch_end(self, args, state, control, **kwargs):
                    self.current_epoch += 1
                    self.trainer.update_progress(self.progress_per_epoch)
                    self.trainer.logger.log("EPOCH", f"Epoch {self.current_epoch}/{num_epochs} completed")

                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs is not None and 'loss' in logs:
                        self.loss_history.append(logs['loss'])
                        if 'eval_accuracy' in logs:
                            self.accuracy_history.append(logs['eval_accuracy'])

                        # Add learning rate tracking
                        if 'learning_rate' in logs:
                            learning_rate = logs['learning_rate']

                        # Update the loss plot
                        self.trainer.window.update_loss_plot(self.loss_history)
            
                        # Calculate running averages
                        recent_losses = self.loss_history[-self.window_size:]
                        avg_loss = sum(recent_losses) / len(recent_losses)
                        total_epochs = args.num_train_epochs
            
                        avg_accuracy = None
                        if self.accuracy_history:
                            recent_accuracy = self.accuracy_history[-self.window_size:]
                            avg_accuracy = sum(recent_accuracy) / len(recent_accuracy)
            
                        self.trainer.logger.log("TRAINING", f"Loss: {logs['loss']:.4f}")
                        self.trainer.window.update_training_metrics(
                            epoch=state.epoch,
                            loss=avg_loss,
                            accuracy=avg_accuracy,
                            learning_rate=learning_rate,
                            total_epochs=total_epochs
                        )

                def on_train_begin(self, args, state, control, **kwargs):
                    self.trainer.logger.log("TRAINING", "Training started")

                def on_train_end(self, args, state, control, **kwargs):
                    self.trainer.logger.log("TRAINING", "Training completed")

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                callbacks=[ProgressCallback(self)]
            )

            trainer.train()
            self.logger.log("INFO", "Training complete!")
            self.update_progress(100)  # Ensure we reach 100%
            
        except Exception as e:
            import traceback
            self.logger.log("ERROR", f"Training failed: {str(e)}\n{traceback.format_exc()}")
