# Federated LibriSpeech: HuBERT Pretraining and Distillation

A comprehensive federated learning framework for training HuBERT models on speech data using the Flower framework. This project implements federated pretraining and knowledge distillation of HuBERT models across distributed clients while preserving data privacy.

## 🏗️ Project Structure

```
federated_librispeech/
├── src/                                    # Main source code directory
│   ├── federated_hubert_pretraining.py     # Federated HuBERT pretraining implementation
│   ├── federated_hubert_distillation.py    # Federated knowledge distillation
│   ├── partition_data.py                   # Dataset partitioning for federated learning
│   ├── generate_kmeans_targets.py          # K-means clustering for pseudo-labels
│   ├── run_pretraining.sh                  # Pretraining execution script
│   ├── run_distillation.sh                 # Distillation execution script
│   ├── run_kmeans.sh                       # K-means generation script
│   ├── configs/                            # Configuration files
│   │   ├── pretraining_config.yaml         # Pretraining configuration
│   │   ├── distillation_config.yaml        # Distillation configuration
│   │   └── client_configs/                 # Client-specific configurations
│   ├── utils/                              # Utility functions
│   │   └── resource_optimizer.py           # Resource optimization utilities
│   ├── federated_librispeech/              # Federated learning data
│   │   └── data/                           # Partitioned dataset storage
│   ├── checkpoints/                        # Model checkpoints
│   └── logs/                               # Training and execution logs
├── speech_commands/                        # Speech Commands dataset
│   ├── testing_list.txt                    # Test set file list
│   ├── validation_list.txt                 # Validation set file list
│   ├── speech_commands_v0.02.tar.gz       # Dataset archive
│   └── [command directories]/              # Audio files organized by command
├── LibriSpeechTars/                        # LibriSpeech dataset archives
│   ├── train-clean-100.tar.gz             # Clean training data (100 hours)
│   ├── train-clean-360.tar.gz             # Clean training data (360 hours)
│   ├── train-other-500.tar.gz             # Other training data (500 hours)
│   ├── dev-other.tar.gz                   # Development set
│   └── test-other.tar.gz                  # Test set
├── s3prl/                                  # S3PRL framework for downstream tasks
├── flvenv/                                 # Python virtual environment
├── setup.md                                # Setup and execution instructions
└── package.json                            # Node.js dependencies (for Claude integration)
```

## 🎯 Project Overview

This project implements a federated learning system for training HuBERT (Hidden-Unit BERT) models on speech data. The system consists of three main components:

1. **Federated Pretraining**: Distributed training of HuBERT models across multiple clients
2. **Knowledge Distillation**: Transfer learning from teacher to student models
3. **Data Partitioning**: Non-IID data distribution for realistic federated scenarios

### Key Features

- **Privacy-Preserving**: Data never leaves client devices
- **Scalable**: Supports multiple clients with different data distributions
- **Efficient**: Optimized for memory and computational resources
- **Flexible**: Configurable training parameters and model architectures

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA-compatible GPU (recommended)
- Access to LibriSpeech dataset
- Flower framework

### Environment Setup

1. **Activate Virtual Environment**:
   ```bash
   source flvenv/bin/activate
   ```

2. **Load Required Modules**:
   ```bash
   module load StdEnv/2023
   module load scipy-stack/2025a
   ```

3. **Navigate to Source Directory**:
   ```bash
   cd src
   ```

### Basic Workflow

1. **Data Partitioning**:
   ```bash
   python partition_data.py
   ```

2. **Generate K-means Targets**:
   ```bash
   python generate_kmeans_targets.py
   ```

3. **Run Federated Pretraining**:
   ```bash
   ./run_pretraining.sh
   ```

4. **Run Knowledge Distillation**:
   ```bash
   ./run_distillation.sh
   ```

## 📁 Detailed Directory Documentation

### `src/` - Main Source Code

#### Core Training Scripts

- **`federated_hubert_pretraining.py`**: Implements federated HuBERT pretraining using Flower framework
  - HuBERT-like transformer architecture with sinusoidal positional encoding
  - Frame-level masking and K-means pseudo-labels
  - FedAdam aggregation strategy
  - Memory-optimized training with gradient checkpointing

- **`federated_hubert_distillation.py`**: Implements federated knowledge distillation
  - Teacher-student architecture with frozen teacher models
  - Combines K-means targets with teacher predictions
  - Frame-level processing with masking
  - Consistent with pretraining implementation

#### Data Processing

- **`partition_data.py`**: LibriSpeech dataset partitioning for federated learning
  - Speaker-based non-IID partitioning
  - Optimized for HuBERT training requirements
  - Multi-process audio processing
  - Memory-efficient data handling

- **`generate_kmeans_targets.py`**: Generates per-client K-means clustering targets
  - Client-specific clustering for privacy preservation
  - MFCC feature extraction with HuBERT-compatible parameters
  - Frame-level target generation
  - Supports ragged array outputs

#### Configuration and Utilities

- **`configs/`**: YAML configuration files for different training modes
  - Model architecture parameters
  - Training hyperparameters
  - Federated learning settings
  - Resource optimization options

- **`utils/resource_optimizer.py`**: Utilities for optimizing computational resources
  - Memory management
  - GPU utilization
  - Batch size optimization

### `speech_commands/` - Downstream Task Dataset

Contains the Speech Commands dataset for fine-tuning and evaluation:
- 35 different voice commands
- Audio files organized by command type
- Training, validation, and test splits
- Used for downstream task evaluation

### `LibriSpeechTars/` - Main Training Dataset

Contains LibriSpeech dataset archives:
- **train-clean-100**: 100 hours of clean speech (training)
- **train-clean-360**: 360 hours of clean speech (training)
- **train-other-500**: 500 hours of other speech (training)
- **dev-other**: Development set for validation
- **test-other**: Test set for evaluation

### `s3prl/` - S3PRL Framework

Integration with S3PRL for downstream task evaluation:
- Fine-tuning on speech commands
- Model evaluation and testing
- Custom upstream model support

## ⚙️ Configuration

### Pretraining Configuration (`configs/pretraining_config.yaml`)

Key parameters:
- **Model**: Hidden size, layers, attention heads
- **Training**: Batch size, learning rate, epochs
- **Data**: Audio length, sample rate, masking
- **Federated**: Number of rounds, client fractions
- **Optimization**: Gradient checkpointing, mixed precision

### Distillation Configuration (`configs/distillation_config.yaml`)

Similar structure with distillation-specific parameters:
- Teacher model paths
- Distillation loss weights
- Student model configurations

## 🔧 Execution Scripts

### `run_pretraining.sh`
Executes federated HuBERT pretraining with proper resource allocation and environment setup.

### `run_distillation.sh`
Runs federated knowledge distillation using pretrained models.

### `run_kmeans.sh`
Generates K-means targets for all clients.

## 📊 Monitoring and Logging

- **Logs**: Comprehensive logging in `src/logs/`
- **Checkpoints**: Model snapshots in `src/checkpoints/`
- **Metrics**: Training metrics and evaluation results
- **Resource Usage**: Memory and GPU utilization tracking

## 🚀 Resource Allocation

For HPC environments, use:
```bash
salloc --account=provider-account --time=8:00:00 --mem=128G --cpus-per-task=16 --gres=gpu:2
```

## 🔍 Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or enable gradient checkpointing
2. **Data Loading**: Verify dataset paths and partitioning
3. **GPU Issues**: Check CUDA compatibility and memory availability

### Log Files

- Partitioning logs: `src/logs/partition_logs/`
- Training logs: `src/logs/pretraining/`
- Distillation logs: `src/logs/distillation/`

## 📚 References

- [HuBERT Paper](https://arxiv.org/abs/2106.07447)
- [Flower Framework](https://flower.dev/)
- [LibriSpeech Dataset](https://www.openslr.org/12/)
- [S3PRL Framework](https://github.com/s3prl/s3prl)

## 🤝 Contributing

This project is designed for research in federated learning for speech processing. Contributions are welcome for:
- Model architecture improvements
- Training optimizations
- Additional datasets
- Performance enhancements

## 📄 License

Please refer to individual component licenses for specific terms.
