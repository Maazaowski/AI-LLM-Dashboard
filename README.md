# AI-LLM-Dashboard

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/Maazaowski/AI-LLM-Dashboard)
[![Build](https://img.shields.io/badge/build-001-green.svg)](https://github.com/Maazaowski/AI-LLM-Dashboard)
[![Last Updated](https://img.shields.io/badge/last%20updated-2024-12-10-lightgrey.svg)](https://github.com/Maazaowski/AI-LLM-Dashboard)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://github.com/Maazaowski/AI-LLM-Dashboard)

A sophisticated dashboard for monitoring and controlling Large Language Model (LLM) training processes with real-time metrics visualization and system monitoring.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Maazaowski/AI-LLM-Dashboard

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python main.py
```

## Features

- **Real-time Training Metrics**
  - Progress tracking
  - Loss visualization
  - Learning rate monitoring
  - Accuracy metrics
  - Training/Validation loss comparison

- **System Monitoring**
  - CPU usage tracking
  - Memory utilization
  - Disk usage statistics
  - Real-time updates

- **Interactive UI Components**
  - Progress bar with training status
  - Live updating graphs
  - Training control buttons (Pause/Stop/Export)
  - Detailed logging console
  - Model selection dropdown

## Technical Requirements

### Core Dependencies
- ttkbootstrap>=1.0.0
- matplotlib>=3.4.0
- psutil>=5.8.0
- pytest>=6.0.0

### Optional Dependencies
- tensorboard>=2.6.0
- pandas>=1.3.0

## Installation Guide

### Prerequisites
- Python >=3.8
- pip (Python package manager)
- Git

### Step-by-step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Maazaowski/AI-LLM-Dashboard
   cd AI-LLM-Dashboard
   ```

2. **Create Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Development Setup

### Running Tests
```bash
pytest tests/
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Document functions and classes

## Troubleshooting Guide

### Common Issues

1. **UI Not Responding**
   - Check system resources
   - Verify training thread status
   - Review log console for errors

2. **Memory Issues**
   - Adjust batch size
   - Monitor system metrics
   - Close unnecessary applications

## Version History

- 1.0.0 (Build 001)
  - Real-time training visualization
  - System metrics monitoring
  - Training control interface

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## Support

For support, please open an issue on the GitHub repository.

---
Last updated: 2024-12-10
Build: 001
