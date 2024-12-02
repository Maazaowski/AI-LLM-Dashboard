import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import scrolledtext
from datetime import datetime
import psutil
import threading
import time
from threading import Thread
from src.ui.styles import UI_STYLES
from src.config.training_config import get_training_args

class TrainingProgressWindow:
    def __init__(self, title="LLM Training Dashboard"):
        self.window = self._setup_window(title)
        self._setup_layout()
        self.running = True

    def _setup_window(self, title):
        window = ttk.Window(title=title, themename="cyborg")
        window.geometry("1024x768")
        window.minsize(800, 600)
        window.grid_columnconfigure(0, weight=1)
        window.grid_rowconfigure(0, weight=1)
        
        # Make window responsive
        window.bind('<Configure>', self._on_window_resize)
        return window

    def _on_window_resize(self, event):
        if not hasattr(self, 'is_resizing'):
            self.is_resizing = False
        
        if not self.is_resizing:
            self.is_resizing = True
            self.window.after(100, self._handle_resize)

    def _handle_resize(self):
        self.window.update_idletasks()
        self.is_resizing = False

    def _setup_layout(self):
        # Create main container with grid layout
        self.main_container = ttk.Frame(self.window, bootstyle="dark")
        self.main_container.pack(fill=BOTH, expand=YES)

        # Header Section
        self._create_header()
        
        # Create left sidebar
        self._create_sidebar()
        
        # Main content area
        self._create_main_content()

        # Create right sidebar for model details
        self._create_right_sidebar()
        
        # Footer
        self._create_footer()

    def _create_header(self):
        header = ttk.Frame(self.main_container, bootstyle="dark")
        header.pack(fill=X, padx=20, pady=10)

        title = ttk.Label(
            header,
            text="LLM Training Dashboard",
            font=UI_STYLES['fonts']['header'],
            bootstyle="info"
        )
        title.pack(side=LEFT)

        # API Status indicator
        self.api_status = ttk.Label(
            header,
            text="● Connected",
            font=UI_STYLES['fonts']['subtitle'],
            bootstyle="success"
        )
        self.api_status.pack(side=RIGHT)

    def _create_sidebar(self):
        sidebar = ttk.Frame(self.main_container, bootstyle="dark")
        sidebar.pack(side=LEFT, fill=Y, padx=20, pady=10)

        # Navigation buttons
        nav_buttons = ["Home", "Train New Model", "View Metrics", "Logs"]
        for btn_text in nav_buttons:
            btn = ttk.Button(
                sidebar,
                text=btn_text,
                bootstyle="info-outline",
                width=20
            )
            btn.pack(pady=5)

        # Model controls
        ttk.Label(sidebar, text="Model Type", bootstyle="info").pack(pady=(20,5))
        ttk.Combobox(sidebar, values=["GPT", "BERT", "T5"]).pack(fill=X)

        ttk.Label(sidebar, text="Dataset Size", bootstyle="info").pack(pady=(20,5))
        ttk.Entry(sidebar).pack(fill=X)

    def _create_main_content(self):
        content = ttk.Frame(self.main_container, bootstyle="dark")
        content.pack(side=LEFT, fill=BOTH, expand=YES, padx=20, pady=10)

        # Create single column layout
        main_column = ttk.Frame(content)
        main_column.pack(fill=BOTH, expand=YES)

        # Add progress bar at the top
        progress_frame = ttk.Frame(main_column, bootstyle="dark")
        progress_frame.pack(fill=X, pady=10)
    
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            bootstyle="info-striped",
            length=900,
            mode="determinate"
        )
        self.progress_bar.pack(fill=X)

        # Training metrics cards
        self._create_metrics_cards(main_column)

        # Move system metrics to right column
        self.create_system_metrics(main_column)
    
        # Graphs section
        self._create_graphs_section(main_column)
    
        # Log console
        self._create_log_console(main_column)

    def _create_metrics_cards(self, parent):

        metrics_frame = ttk.Frame(parent, bootstyle="dark")
        metrics_frame.pack(fill=X, pady=10)

        # Initialize metric label references
        self.epoch_label = None
        self.training_loss_label = None 
        self.validation_loss_label = None
        self.accuracy_label = None
        self.learning_rate_label = None

        # Create metric cards with stored references
        metrics = [
            ("Epochs", "0/0", "epoch_label"),
            ("Training Loss", "0.000", "training_loss_label"),
            ("Validation Loss", "0.000", "validation_loss_label"), 
            ("Accuracy", "0%", "accuracy_label"),
            ("Learning Rate", "0.000", "learning_rate_label")
        ]

        # Create metric cards in a horizontal layout
        for label, initial_value, attr_name in metrics:
            card = ttk.LabelFrame(
                metrics_frame,
                text=label,
                bootstyle="info",
                padding=10
            )
            card.pack(side=LEFT, fill=X, expand=True, padx=5)
            
            # Create and store label reference
            metric_label = ttk.Label(
                card,
                text=initial_value,
                font=UI_STYLES['fonts']['title'],
                bootstyle="info"
            )
            metric_label.pack()
            
            # Store reference as class attribute
            setattr(self, attr_name, metric_label)

    def _create_right_sidebar(self):

        self._create_model_details()

    def _create_footer(self):
        footer = ttk.Frame(self.main_container, bootstyle="dark")
        footer.pack(fill=X, side=BOTTOM, padx=20, pady=10)
        
        button_frame = ttk.Frame(footer, bootstyle="dark")
        button_frame.pack(expand=True)
        
        button_style = {'width': 15, 'padding': 10}
        
        ttk.Button(
            button_frame,
            text="Pause Training",
            bootstyle="warning",  # Removed -outline for consistency
            command=self.suspend_training,
            **button_style
        ).pack(side=LEFT, padx=10)

        ttk.Button(
            button_frame,
            text="Stop Training",
            bootstyle="danger-outline",
            command=self.terminate_training,
            **button_style
        ).pack(side=LEFT, padx=10)

        ttk.Button(
            button_frame,
            text="Export Metrics",
            bootstyle="info-outline",
            command=self.export_metrics,
            **button_style
        ).pack(side=LEFT, padx=10)

    def export_metrics(self):
        self.update_log("SYSTEM", "Exporting training metrics...")
        # Add export functionality here

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

    def create_system_metrics(self, parent):
        # Create system metrics frame in the metrics dashboard
        system_frame = ttk.LabelFrame(
            parent,
            text="System Metrics",
            bootstyle="info",
            padding=10
        )
        system_frame.pack(fill=X, pady=5)

        self.cpu_label = ttk.Label(system_frame, text="CPU Usage: 0%")
        self.cpu_label.pack(anchor=W)
        
        self.memory_label = ttk.Label(system_frame, text="Memory Usage: 0%")
        self.memory_label.pack(anchor=W)
        
        self.disk_label = ttk.Label(system_frame, text="Disk Usage: 0%")
        self.disk_label.pack(anchor=W)

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

    def update_training_metrics(self, epoch=0, loss=0.0, accuracy=None, learning_rate=0.0, total_epochs=3):
        # Update labels with current metrics
        formatted_epoch = f"{epoch:.2f}/{total_epochs}"
        self.epoch_label.config(text=formatted_epoch)
        self.training_loss_label.config(text=f"{loss:.4f}")
    
        # Handle accuracy which might be None
        accuracy_text = f"{accuracy:.2f}%" if accuracy is not None else "N/A"
        self.accuracy_label.config(text=accuracy_text)
        
        self.learning_rate_label.config(text=f"{learning_rate:.4f}")

    def _create_graphs_section(self, parent):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        graphs_frame = ttk.LabelFrame(parent, text="Training Progress", bootstyle="info", padding=10)
        graphs_frame.pack(fill=X, pady=10)

        # Create figure with dark theme styling
        self.fig = Figure(figsize=(8, 4), facecolor=UI_STYLES['colors']['card_bg'])
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(UI_STYLES['colors']['card_bg'])
    
        # Style the plot
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')

        # Initialize empty plots with initial range 0-100
        self.current_step = 0
        self.ax.set_xlim(0, 100)
        self.loss_line, = self.ax.plot([], [], 'c-', label='Training Loss', linewidth=2)
        self.avg_loss_line, = self.ax.plot([], [], 'r--', label='Moving Average', linewidth=2)
    
        self.ax.set_xlabel('Steps')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Training Loss Over Time')
        self.ax.grid(True, color='gray', alpha=0.2)
        self.ax.legend(facecolor=UI_STYLES['colors']['card_bg'], labelcolor='white')

        self.canvas = FigureCanvasTkAgg(self.fig, master=graphs_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    def update_loss_plot(self, losses, window_size=50):
        if losses:
            self.current_step += 1
            steps = list(range(self.current_step))
        
            # Calculate moving average
            moving_avg = []
            for i in range(len(losses)):
                start_idx = max(0, i - window_size + 1)
                moving_avg.append(sum(losses[start_idx:i+1]) / (i - start_idx + 1))

            # Update plot data
            self.loss_line.set_data(steps, losses)
            self.avg_loss_line.set_data(steps, moving_avg)
        
            # Update x-axis range if current_step exceeds current limit
            x_min, x_max = self.ax.get_xlim()
            if self.current_step >= x_max:
                self.ax.set_xlim(0, x_max + 100)
        
            # Update y-axis limits
            self.ax.relim()
            self.ax.autoscale_view(scalex=False)  # Only autoscale y-axis
        
            # Refresh canvas
            self.canvas.draw()
    def _create_log_console(self, parent):
        # Create log console frame
        log_frame = ttk.LabelFrame(parent, text="Training Logs", bootstyle="info", padding=10)
        log_frame.pack(fill=BOTH, expand=YES, pady=10)

        # Create scrolled text widget for logs
        self.log_box = scrolledtext.ScrolledText(
            log_frame,
            width=90,
            height=15,
            font=UI_STYLES['fonts']['content'],
            bg=UI_STYLES['colors']['card_bg'],
            fg=UI_STYLES['colors']['text_primary']
        )
        self.log_box.pack(fill=BOTH, expand=YES)

        # Add initial log message
        self.update_log("SYSTEM", "Training console initialized and ready")


    def _create_model_details(self):
        details_frame = ttk.LabelFrame(self.main_container, text="Model Details", bootstyle="info", padding=10)
        details_frame.pack(side=RIGHT, fill=Y, padx=20, pady=10)

        # Create a grid for model information
        details = [
            ("Model Name:", "LLM_Version1"),
            ("Dataset Used:", "Dataset_XYZ"),
            ("Start Time:", "10:00 AM, Nov 27"),
            ("Estimated Completion:", "12:30 PM, Nov 27")
        ]

        for label, value in details:
            row_frame = ttk.Frame(details_frame)
            row_frame.pack(fill=X, pady=5)
            
            ttk.Label(
                row_frame, 
                text=label,
                font=UI_STYLES['fonts']['subtitle'],
                bootstyle="info"
            ).pack(side=LEFT, padx=10)
            
            ttk.Label(
                row_frame,
                text=value,
                font=UI_STYLES['fonts']['content']
            ).pack(side=LEFT, padx=10)


