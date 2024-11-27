import threading
from src.ui.training_window import TrainingProgressWindow
from src.training.trainer import ModelTrainer
from src.utils.logger import TrainingLogger
from src.config.training_config import get_training_args

def start_training(progress_window):
    #logger = TrainingLogger(progress_window)
    trainer = ModelTrainer(progress_window)
    training_args = get_training_args()
    
    # Run training in a separate thread
    training_thread = threading.Thread(
        target=trainer.train,
        args=(training_args,),
        daemon=True
    )
    training_thread.start()

if __name__ == "__main__":
    # Initialize the progress window
    progress_window = TrainingProgressWindow()
    
    # Start the training process
    start_training(progress_window)
    
    # Start the Tkinter event loop
    progress_window.window.mainloop() 