from datetime import datetime
from typing import Protocol

class LoggerInterface(Protocol):
    def update_log(self, category: str, message: str) -> None: ...
    def update_progress(self, progress_value: float) -> None: ...

class TrainingLogger:
    def __init__(self, ui_logger: LoggerInterface):
        self.ui_logger = ui_logger

    def log(self, category: str, message: str):
        self.ui_logger.update_log(category, message)

    def update_progress(self, progress: float):
        progress = max(0, min(100, progress))
        self.ui_logger.update_progress(progress) 