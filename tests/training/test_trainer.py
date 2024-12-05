import unittest
from unittest.mock import MagicMock, patch
from src.training.trainer import ModelTrainer
from transformers import TrainingArguments
import warnings

# Add this at the top of your test file
warnings.filterwarnings("ignore", category=DeprecationWarning)

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        self.mock_window = MagicMock()
        self.trainer = ModelTrainer(self.mock_window)
    
    def test_progress_tracking(self):
        """Test progress updates during training"""
        initial_progress = self.trainer.current_progress
        self.trainer.update_progress(25.0)
        self.assertEqual(self.trainer.current_progress, initial_progress + 25.0)
    
    @patch('src.training.trainer.load_dataset')
    def test_dataset_loading(self, mock_load_dataset):
        """Test dataset loading functionality"""
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = MagicMock()
        mock_load_dataset.return_value = mock_dataset
        
        self.trainer.load_dataset()
        mock_load_dataset.assert_called_once()
    
    @patch('src.training.trainer.GPT2LMHeadModel')
    def test_model_initialization(self, mock_gpt2):
        """Test model initialization"""
        self.trainer.initialize_model()
        mock_gpt2.from_pretrained.assert_called_once_with("gpt2")

    def test_training_args_integration(self):
        """Test training arguments integration"""
        args = TrainingArguments(
            output_dir="./test_output",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8
        )
        
        with patch('src.training.trainer.Trainer') as mock_trainer:
            self.trainer.train(args)
            mock_trainer.assert_called()

    def test_logger_initialization(self):
        """Test logger initialization"""
        self.assertIsNotNone(self.trainer.logger)
        self.assertEqual(self.trainer.current_progress, 0)

    def test_tokenizer_initialization(self):
        """Test tokenizer initialization"""
        self.assertIsNotNone(self.trainer.tokenizer)

    def test_progress_limits(self):
        """Test progress update boundaries"""
        self.trainer.current_progress = 0
        self.trainer.update_progress(150)
        self.assertEqual(self.trainer.current_progress, 100)
    
