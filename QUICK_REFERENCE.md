# Quick Reference Guide: Federated LibriSpeech

## 🚀 Essential Commands

### Environment Setup
```bash
# Activate virtual environment
source flvenv/bin/activate

# Load required modules
module load StdEnv/2023
module load scipy-stack/2025a

# Navigate to source directory
cd src
```

### Resource Allocation (HPC)
```bash
# Basic allocation
salloc --account=def-aravila --time=4:00:00 --mem=64G --cpus-per-task=8 --gres=gpu:1

# Extended allocation
salloc --account=def-aravila --time=8:00:00 --mem=128G --cpus-per-task=16 --gres=gpu:2
```

### TMUX Session Management
```bash
# Create new session
tmux new-session -s training

# Detach from session
tmux detach

# Attach to existing session
tmux attach -t training

# List sessions
tmux ls

# Kill session
tmux kill-session -t training
```

## 📁 File Structure Quick Reference

```
src/
├── federated_hubert_pretraining.py     # Main pretraining script
├── federated_hubert_distillation.py    # Main distillation script
├── partition_data.py                   # Dataset partitioning
├── generate_kmeans_targets.py          # K-means target generation
├── run_pretraining.sh                  # Pretraining execution
├── run_distillation.sh                 # Distillation execution
├── run_kmeans.sh                       # K-means execution
├── configs/                            # Configuration files
├── utils/                              # Utility functions
├── checkpoints/                        # Model checkpoints
└── logs/                               # Training logs
```

## ⚙️ Configuration Quick Reference

### Key Pretraining Parameters
```yaml
pretraining:
  hidden_size: 768              # Model hidden dimension
  num_hidden_layers: 12         # Number of transformer layers
  batch_size: 8                 # Training batch size
  learning_rate: 5e-4           # Learning rate
  num_rounds: 2                 # Federated rounds
  min_fit_clients: 2            # Minimum clients per round
```

### Key Distillation Parameters
```yaml
distillation:
  teacher_model_path: "path/to/teacher"
  alpha: 0.5                    # Distillation loss weight
  temperature: 2.0              # Softmax temperature
  student_hidden_size: 384      # Student model size
```

## 🔧 Common Operations

### 1. Data Partitioning
```bash
# Basic partitioning
python partition_data.py

# With custom config
python partition_data.py --config configs/custom_partition.yaml

# Check partitioning logs
tail -f logs/partition_logs/partitioning.log
```

### 2. K-means Target Generation
```bash
# Generate targets for all clients
python generate_kmeans_targets.py

# With custom parameters
python generate_kmeans_targets.py --n_clusters 256 --max_length 30000

# Check generation logs
tail -f utils/kmeans_generation.log
```

### 3. Federated Pretraining
```bash
# Run pretraining
./run_pretraining.sh

# Check training logs
tail -f logs/pretraining/federated_pretraining.log

# Monitor checkpoints
ls -la checkpoints/pretraining/server/
```

### 4. Knowledge Distillation
```bash
# Run distillation
./run_distillation.sh

# Check distillation logs
tail -f logs/distillation/federated_distillation.log

# Monitor student model progress
ls -la checkpoints/distillation/students/
```

### 5. Downstream Evaluation
```bash
# Fine-tune on speech commands
python s3prl/run_downstream.py -m train -u hubert_local \
    -k checkpoints/pretraining/server/best_global_model.pt \
    -d speech_commands -n ExpName

# Evaluate model
python s3prl/run_downstream.py -m evaluate \
    -e result/downstream/ExpName/dev-best.ckpt
```

## 📊 Monitoring and Debugging

### Check Training Progress
```bash
# Monitor server logs
tail -f logs/pretraining/server/server.log

# Monitor client logs
tail -f logs/pretraining/clients/client_0.log

# Check metrics file
cat logs/pretraining/federated_pretraining_metrics.json
```

### Resource Monitoring
```bash
# Check GPU usage
nvidia-smi

# Check memory usage
free -h

# Check disk space
df -h

# Check process status
ps aux | grep python
```

### Common Log Patterns
```bash
# Find error messages
grep -i "error" logs/pretraining/*.log

# Find memory issues
grep -i "memory\|oom" logs/pretraining/*.log

# Find training metrics
grep -i "loss\|accuracy" logs/pretraining/*.log
```

## 🐛 Troubleshooting Quick Fixes

### Memory Issues
```bash
# Reduce batch size in config
sed -i 's/batch_size: [0-9]*/batch_size: 4/' configs/pretraining_config.yaml

# Enable gradient checkpointing
sed -i 's/gradient_checkpointing: false/gradient_checkpointing: true/' configs/pretraining_config.yaml

# Reduce number of workers
sed -i 's/num_workers: [0-9]*/num_workers: 4/' configs/pretraining_config.yaml
```

### GPU Issues
```bash
# Check CUDA version compatibility
python -c "import torch; print(torch.version.cuda)"

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Set memory fraction
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### Data Loading Issues
```bash
# Verify dataset paths
ls -la federated_librispeech/data/

# Check manifest files
head -5 federated_librispeech/data/client_0/manifest.csv

# Verify audio files exist
find federated_librispeech/data/ -name "*.flac" | head -5
```

## 📈 Performance Optimization

### Memory Optimization
```yaml
# In config file
gradient_checkpointing: true
mixed_precision: true
pin_memory: false
num_workers: 4
prefetch_factor: 1
```

### Training Optimization
```yaml
# In config file
gradient_accumulation_steps: 4
max_grad_norm: 1.0
weight_decay: 0.01
dropout: 0.1
```

### Federated Learning Optimization
```yaml
# In config file
fraction_fit: 1.0
fraction_evaluate: 1.0
min_available_clients: 2
local_epochs: 1
```

## 🔍 Debugging Commands

### Model Inspection
```python
# Check model parameters
python -c "
import torch
from federated_hubert_pretraining import HubertBase
model = HubertBase()
print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
"
```

### Data Inspection
```python
# Check dataset statistics
python -c "
import pandas as pd
df = pd.read_csv('federated_librispeech/data/client_0/manifest.csv')
print(f'Client 0 has {len(df)} samples')
print(f'Total duration: {df.duration.sum():.2f} seconds')
"
```

### Configuration Validation
```python
# Validate YAML config
python -c "
import yaml
with open('configs/pretraining_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print('Configuration loaded successfully')
print(f'Model hidden size: {config[\"pretraining\"][\"hidden_size\"]}')
"
```

## 📚 Useful Aliases

Add these to your `~/.bashrc`:
```bash
# Quick navigation
alias fed='cd /home/saadan/scratch/federated_librispeech'
alias src='cd /home/saadan/scratch/federated_librispeech/src'
alias logs='cd /home/saadan/scratch/federated_librispeech/src/logs'

# Environment setup
alias activate='source flvenv/bin/activate && module load StdEnv/2023 && module load scipy-stack/2025a'

# Quick monitoring
alias monitor='watch -n 1 "nvidia-smi && echo && free -h"'
alias check_logs='tail -f logs/pretraining/federated_pretraining.log'
```

## 🚨 Emergency Commands

### Stop All Training
```bash
# Kill all Python processes
pkill -f "federated_hubert"

# Kill specific training processes
pkill -f "pretraining\|distillation"

# Force kill if needed
pkill -9 -f "federated_hubert"
```

### Reset Environment
```bash
# Deactivate virtual environment
deactivate

# Unload modules
module purge

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"
```

### Clean Temporary Files
```bash
# Remove temporary files
rm -rf src/__pycache__
rm -rf src/utils/__pycache__
rm -rf src/federated_librispeech/__pycache__

# Clear logs (be careful!)
# rm -rf src/logs/*
```
