import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import scrolledtext
from datetime import datetime
import psutil
import threading
import time
from threading import Thread

class TrainingProgressWindow:
    def __init__(self, title="AI Neural Network Training Interface"):
        self.window = self._setup_window(title)
        self.running = True
        self._setup_ui_components()
        self.update_system_metrics()

    def _setup_window(self, title):
        window = ttk.Window(
            title=title,
            themename="cyborg",
            resizable=(False, False)
        )
        
        window_width = 950
        window_height = 700
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        position_top = int(screen_height / 2 - window_height / 2)
        position_left = int(screen_width / 2 - window_width / 2)
        
        window.geometry(f'{window_width}x{window_height}+{position_left}+{position_top}')
        return window

    def _setup_ui_components(self):
        # Main container
        main_frame = ttk.Frame(self.window, bootstyle="dark")
        main_frame.pack(fill=BOTH, expand=YES, padx=20, pady=20)

        # Holographic title
        title_frame = ttk.Frame(main_frame, bootstyle="dark")
        title_frame.pack(fill=X, pady=10)
        
        title_label = ttk.Label(
            title_frame,
            text="QUANTUM AI TRAINING MATRIX",
            font=("Helvetica", 24, "bold"),
            bootstyle="info"
        )
        title_label.pack()

        # Metrics dashboard
        metrics_frame = ttk.Frame(main_frame, bootstyle="dark")
        metrics_frame.pack(fill=X, pady=10)
        
        # Stats display
        self.stats_frame = ttk.LabelFrame(
            metrics_frame,
            text="TRAINING METRICS",
            bootstyle="info",
            padding=10
        )
        self.stats_frame.pack(fill=X, pady=5)

        # Create a grid of metric displays
        self.epoch_label = ttk.Label(
            self.stats_frame,
            text="EPOCHS: 0/100",
            font=("Helvetica", 12, "bold"),
            bootstyle="info"
        )
        self.epoch_label.grid(row=0, column=0, padx=10)

        self.loss_label = ttk.Label(
            self.stats_frame,
            text="LOSS: 0.000",
            font=("Helvetica", 12, "bold"),
            bootstyle="warning"
        )
        self.loss_label.grid(row=0, column=1, padx=10)

        self.accuracy_label = ttk.Label(
            self.stats_frame,
            text="ACCURACY: 0.00%",
            font=("Helvetica", 12, "bold"),
            bootstyle="success"
        )
        self.accuracy_label.grid(row=0, column=2, padx=10)

        # Advanced log console using tkinter's scrolledtext
        self.log_box = scrolledtext.ScrolledText(
            main_frame,
            width=90,
            height=20,
            font=("Consolas", 12),
            bg='#2b2b2b',
            fg='#00ff9f'
        )
        self.log_box.pack(fill=BOTH, expand=YES, pady=10)

        # Futuristic progress bar
        progress_frame = ttk.Frame(main_frame, bootstyle="dark")
        progress_frame.pack(fill=X, pady=10)
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            bootstyle="info-striped",
            length=900,
            mode="determinate"
        )
        self.progress_bar.pack(fill=X)

        # Control panel
        control_frame = ttk.Frame(main_frame, bootstyle="dark")
        control_frame.pack(fill=X, pady=10)
        
        self.suspend_button = ttk.Button(
            control_frame,
            text="Suspend",
            command=self.suspend_training,
            bootstyle="warning"
        )
        self.suspend_button.pack(side=RIGHT, padx=5)
        self.terminate_button = ttk.Button(
              control_frame,
              text="Terminate",
              command=self.terminate_training,
              bootstyle="danger"
          )
        self.terminate_button.pack(side=RIGHT, padx=5)

        self.update_system_metrics()

    def close(self):
          """Close the window and destroy the Tkinter instance"""
          self.window.quit()
          self.window.destroy()

    def update_log(self, category, message):
          timestamp = datetime.now().strftime("%H:%M:%S")
          log_entry = f"[{timestamp}] [{category}] {message}\n"
          self.log_box.insert(END, log_entry)
          self.log_box.see(END)
    def update_progress(self, progress_value: float):
        """Update the progress bar value"""
        self.progress_bar['value'] = progress_value
        self.window.update_idletasks()

    def show_completion_message(self):
        """Display completion message and set progress to 100%"""
        self.update_log("INFO", "Training Complete! Press close button to exit.")
        self.update_progress(100)

    def show_error_message(self, error_message: str):
        """Display error message and reset progress"""
        self.update_log("ERROR", f"Error: {error_message}")
        self.update_progress(0)

    def update_tokenization_progress(self, current, total):
        progress_percentage = (current / total) * 100
        self.progress_bar['value'] = progress_percentage
        self.window.update_idletasks()  # Force UI update
    
    def log_tokenization_status(self, message):
        self.update_log("Tokenization", message)

    def suspend_training(self):
        self.training_suspended = True
        self.update_log("System", "Training process suspended")
    
    def terminate_training(self):
        self.training_terminated = True
        self.update_log("System", "Training process terminated")
        self.window.destroy()

    def update_system_metrics(self):
        # Create system metrics frame in the metrics dashboard
        system_frame = ttk.LabelFrame(
            self.stats_frame,
            text="SYSTEM METRICS",
            bootstyle="info",
            padding=10
        )
        system_frame.grid(row=1, column=0, columnspan=3, sticky='ew', pady=5)  # Using grid instead of pack

        # Create labels for different metrics
        self.cpu_label = ttk.Label(system_frame, text="CPU Usage: 0%")
        self.cpu_label.grid(row=0, column=0, sticky=W)
        
        self.memory_label = ttk.Label(system_frame, text="Memory Usage: 0%")
        self.memory_label.grid(row=1, column=0, sticky=W)
        
        self.disk_label = ttk.Label(system_frame, text="Disk Usage: 0%")
        self.disk_label.grid(row=2, column=0, sticky=W)

        def update():
            while self.running:
                try:
                    cpu_percent = psutil.cpu_percent()
                    memory = psutil.virtual_memory()
                    memory_percent = memory.percent
                    
                    # Schedule UI updates on main thread
                    self.window.after(0, lambda: self.cpu_label.configure(text=f"CPU Usage: {cpu_percent}%"))
                    self.window.after(0, lambda: self.memory_label.configure(text=f"Memory Usage: {memory_percent}%"))
                    time.sleep(1)
                except Exception:
                    break
        
        self.running = True
        self.metrics_thread = Thread(target=update, daemon=True)
        self.metrics_thread.start()
    def update_training_metrics(self, epoch=0, loss=0.0, accuracy=None):
        # Update labels with current metrics
        self.epoch_label.config(text=f"Current Epoch: {epoch}")
        self.loss_label.config(text=f"Loss: {loss:.4f}")
    
        # Handle accuracy which might be None
        accuracy_text = f"Accuracy: {accuracy:.2f}%" if accuracy is not None else "Accuracy: N/A"
        self.accuracy_label.config(text=accuracy_text)
