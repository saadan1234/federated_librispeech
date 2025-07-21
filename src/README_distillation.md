# Federated HuBERT Knowledge Distillation

This directory contains the implementation of federated knowledge distillation for HuBERT models on LibriSpeech data. The approach uses a server-side teacher model with client-side student models, where knowledge distillation is performed at the server after parameter aggregation.

## Features

- **Server-side Teacher Model**: Full HuBERT model (768 hidden, 12 layers) maintained on the server
- **Client-side Student Models**: Compact HuBERT models (384 hidden, 3 layers) for efficient training
- **Knowledge Distillation**: Performed at server after FedAvg aggregation
- **Dynamic Resource Utilization**: Automatically optimizes for available GPUs and CPU cores
- **Flower Integration**: Uses Flower framework for federated learning coordination

## Architecture

### Teacher Model (Server)
- Hidden size: 768
- Layers: 12
- Attention heads: 12
- Intermediate size: 3072
- Parameters: ~95M

### Student Model (Client)
- Hidden size: 384 (50% of teacher)
- Layers: 3 (25% of teacher)
- Attention heads: 6 (50% of teacher)
- Intermediate size: 1536 (50% of teacher)
- Parameters: ~12M (87% reduction)

## Files

### Core Implementation
- `federated_hubert_distillation.py`: Main distillation implementation
- `configs/distillation_config.yaml`: Configuration file
- `run_distillation.py`: Python runner script
- `run_distillation.sh`: Shell script for easy execution

### Key Components
- `StudentHuBERTModel`: Compact student model architecture
- `KnowledgeDistillationLoss`: Multi-level distillation loss
- `FederatedHuBERTDistillationClient`: Flower client for student training
- `ServerSideDistillationStrategy`: Custom strategy with server-side distillation

## Usage

### Quick Start
```bash
# Run with default settings (10 clients, 20 rounds)
./run_distillation.sh

# Custom configuration
./run_distillation.sh --num-clients 5 --num-rounds 10
```

### Python Script
```bash
# Run distillation experiment
python run_distillation.py --num-clients 10 --num-rounds 20

# Skip resource optimization
python run_distillation.py --no-optimize
```

### Direct Execution
```bash
# Run with automatic resource optimization
python federated_hubert_distillation.py --config configs/distillation_config.yaml --simulation --num-clients 10 --num-rounds 20

# Skip resource optimization
python federated_hubert_distillation.py --config configs/distillation_config.yaml --simulation --num-clients 10 --num-rounds 20 --no-optimize
```

## Configuration

Key configuration parameters in `configs/distillation_config.yaml`:

### Student Model
```yaml
distillation:
  student_hidden_size: 384
  student_num_layers: 3
  student_num_heads: 6
  student_intermediate_size: 1536
```

### Distillation Parameters
```yaml
distillation:
  temperature: 4.0        # Softmax temperature
  weight: 0.5            # Teacher knowledge blending weight
  num_rounds: 20         # Training rounds
  local_epochs: 1        # Local training epochs
  batch_size: 8          # Batch size per client
  learning_rate: 1e-4    # Learning rate
```

### Resource Optimization
The system automatically detects:
- Available CPU cores
- GPU count and memory
- SLURM allocations
- Optimal batch sizes and worker counts

## Knowledge Distillation Process

1. **Client Training**: Students train locally on self-supervised tasks
2. **Parameter Aggregation**: FedAvg aggregates student parameters
3. **Server-side Distillation**: Teacher knowledge is distilled into aggregated student
4. **Parameter Distribution**: Distilled parameters sent back to clients

### Distillation Loss Components
- **Output Distillation**: KL divergence between teacher and student predictions
- **Feature Distillation**: MSE loss between intermediate representations
- **Layer Mapping**: Student layers 0,1,2 → Teacher layers 3,7,11

## Hardware Requirements

### Minimum
- 4 CPU cores
- 8GB RAM
- 1 GPU (optional, but recommended)

### Recommended
- 16+ CPU cores
- 32GB+ RAM
- 4+ GPUs with 16GB+ memory each

## Output Files

### Logs
- `logs/federated_distillation.log`: Main training log
- `logs/distillation_TIMESTAMP.log`: Timestamped experiment logs

### Checkpoints
- `checkpoints/distillation/`: Periodic model checkpoints
- `checkpoints/distillation/best_model.pt`: Best performing model

### Final Model
- `models/distilled_hubert.pt`: Final distilled student model

## Monitoring

The experiment provides detailed logging of:
- Resource allocation per client
- Training progress per round
- Distillation loss components
- Model performance metrics

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size in config
2. **Slow Training**: Increase number of data workers
3. **Poor Convergence**: Adjust distillation temperature or weight

### Debug Mode
```bash
# Enable debug logging
export PYTHONPATH=/lustre07/scratch/saadan/federated_librispeech/src:$PYTHONPATH
python federated_hubert_distillation.py --config configs/distillation_config.yaml --simulation --num-clients 2 --num-rounds 2
```

## Research Context

This implementation follows the DistilBERT and DistilHuBERT approaches:
- Server-side distillation for federated learning
- Layer-wise knowledge transfer
- Efficient student architecture design
- Dynamic resource optimization

## Dependencies

- PyTorch
- Flower (flwr)
- Transformers
- torchaudio
- soundfile
- pandas
- numpy
- PyYAML

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{federated_hubert_distillation,
  title={Federated HuBERT Knowledge Distillation},
  author={Your Name},
  year={2025},
  howpublished={\\url{https://github.com/your-repo/federated_librispeech}}
}
```