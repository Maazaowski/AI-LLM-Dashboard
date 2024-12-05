import unittest
from src.ui.training_window import TrainingProgressWindow
import tkinter as tk

class TestTrainingWindow(unittest.TestCase):
    def test_window_creation(self):
        window = TrainingProgressWindow()
        self.assertIsNotNone(window)
        window.close()
        
    @classmethod
    def setUpClass(cls):
        cls.window = TrainingProgressWindow()
    
    @classmethod
    def tearDownClass(cls):
        try:
            cls.window.close()
        except:
            pass  # Window might already be destroyed
    
    def test_metrics_update(self):
        """Test if training metrics update correctly"""
        self.window.update_training_metrics(
            epoch=1,
            loss=0.5,
            accuracy=75.5,
            learning_rate=0.001,
            total_epochs=3
        )
        
        self.assertEqual(self.window.epoch_label.cget("text"), "1.00/3")
        self.assertEqual(self.window.training_loss_label.cget("text"), "0.5000")
        self.assertEqual(self.window.accuracy_label.cget("text"), "75.50%")
        self.assertEqual(self.window.learning_rate_label.cget("text"), "0.0010")

    def test_progress_bar(self):
        """Test progress bar updates"""
        test_values = [0, 25, 50, 75, 100]
        for value in test_values:
            self.window.update_progress(value)
            self.assertEqual(self.window.progress_bar['value'], value)

    def test_log_messages(self):
        """Test log message functionality"""
        test_message = "Test training started"
        self.window.update_log("TEST", test_message)
        log_content = self.window.log_box.get("1.0", tk.END)
        self.assertIn(test_message, log_content)

    def test_system_metrics_initialization(self):
        """Test system metrics display initialization"""
        self.assertIsNotNone(self.window.cpu_label)
        self.assertIsNotNone(self.window.memory_label)
        self.assertIsNotNone(self.window.disk_label)

    def test_training_suspension(self):
        """Test training suspension functionality"""
        self.window.suspend_training()
        self.assertTrue(self.window.training_suspended)

    def test_training_termination(self):
        """Test training termination functionality"""
        self.window.terminate_training()
        self.assertTrue(self.window.training_terminated)
