# Developer Guide: Federated LibriSpeech

This document provides detailed technical information for developers working with the Federated LibriSpeech project.

## 🏗️ Architecture Overview

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Flower Server │    │  Client 1       │    │  Client N       │
│                 │    │                 │    │                 │
│ • FedAdam       │◄──►│ • HubertModel  │    │ • HubertModel  │
│ • Aggregation   │    │ • Local Data    │    │ • Local Data    │
│ • Checkpointing │    │ • Training Loop │    │ • Training Loop │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Storage  │    │  K-means       │    │  K-means       │
│                 │    │  Targets       │    │  Targets       │
│ • LibriSpeech   │    │ • Client-      │    │ • Client-      │
│ • Partitioned   │    │   specific     │    │   specific     │
│ • Non-IID       │    │ • MFCC-based   │    │ • MFCC-based   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Core Implementation Details

### 1. Model Architecture (`federated_hubert_pretraining.py`)

#### HubertBase Class
```python
class HubertBase(nn.Module):
    def __init__(self, hidden_size: int = 768, num_layers: int = 12, 
                 vocab_size: int = 504, frame_stride: int = 320):
        # Transformer encoder with positional encoding
        # Frame-level output processing
        # K-means vocabulary integration
```

**Key Features:**
- **Hidden Size**: 768 (configurable)
- **Layers**: 12 transformer encoder layers
- **Vocabulary**: 504 K-means clusters
- **Frame Stride**: 320 samples (20ms at 16kHz)

#### Positional Encoding
```python
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs"""
    def __init__(self, hidden_size: int, max_len: int = 10000):
        # Sinusoidal encoding with learnable parameters
```

### 2. Data Processing Pipeline

#### Audio Processing (`generate_kmeans_targets.py`)
```python
def load_audio(audio_path: Path, sample_rate: int, max_length: int) -> torch.Tensor:
    """Load and preprocess audio for HuBERT training"""
    # Resample to 16kHz
    # Truncate to max_length (default: 40000 samples)
    # Convert to mono if stereo
```

#### MFCC Feature Extraction
```python
def compute_mfcc_frames(audio: torch.Tensor, sample_rate: int, frame_stride: int) -> torch.Tensor:
    """Extract MFCC features compatible with HuBERT"""
    mfcc = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        melkwargs={
            "n_fft": 400,
            "n_mels": 40,
            "hop_length": frame_stride,  # 320
            "win_length": 400,
            "center": False,
        },
    )
```

#### K-means Clustering
```python
def fit_client_kmeans(all_frames: List[torch.Tensor], n_clusters: int, 
                     batch_size: int, max_iter: int, seed: int) -> MiniBatchKMeans:
    """Client-specific K-means for privacy preservation"""
    # MiniBatchKMeans for memory efficiency
    # Client-specific clustering (no data sharing)
    # Adaptive cluster count based on available frames
```

### 3. Federated Learning Implementation

#### Flower Client (`federated_hubert_pretraining.py`)
```python
class HubertClient(NumPyClient):
    def __init__(self, model: HubertBase, train_loader: DataLoader, 
                 val_loader: DataLoader, device: str, config: Dict):
        # Local model and data
        # Training configuration
        # Device management (CPU/GPU)
    
    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict]:
        """Execute local training round"""
        # Update model parameters
        # Train for specified epochs
        # Return updated parameters and metadata
    
    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict]:
        """Evaluate local model performance"""
        # Update model parameters
        # Evaluate on validation set
        # Return loss and metrics
```

#### Server Strategy (`federated_hubert_pretraining.py`)
```python
# FedAdam aggregation strategy
strategy = FedAdam(
    min_fit_clients=config['min_fit_clients'],
    min_evaluate_clients=config['min_evaluate_clients'],
    min_available_clients=config['min_available_clients'],
    fraction_fit=config['fraction_fit'],
    fraction_evaluate=config['fraction_evaluate'],
    initial_parameters=ndarrays_to_parameters(initial_weights),
    eta=config['learning_rate'],
    eta_l=config['local_learning_rate'],
    beta_1=config['beta_1'],
    beta_2=config['beta_2'],
    tau=config['tau']
)
```

### 4. Data Partitioning (`partition_data.py`)

#### Speaker-Based Partitioning
```python
def partition_by_speaker(dataset_info: pd.DataFrame, num_clients: int, 
                        partition_method: str = 'speaker_based') -> Dict[int, List[str]]:
    """Create non-IID partitions based on speaker identity"""
    # Group by speaker ID
    # Distribute speakers across clients
    # Ensure balanced partition sizes
```

#### Audio Processing Pipeline
```python
def process_audio_file(audio_file: Path) -> Optional[Dict[str, Any]]:
    """Extract metadata from audio files"""
    # Duration calculation
    # File path resolution
    # Error handling for corrupted files
```

## 📊 Configuration Management

### YAML Configuration Structure

#### Pretraining Configuration (`configs/pretraining_config.yaml`)
```yaml
pretraining:
  # Model architecture
  hidden_size: 768
  num_attention_heads: 12
  num_hidden_layers: 12
  vocab_size: 504
  
  # Training parameters
  batch_size: 8
  local_epochs: 1
  learning_rate: 5e-4
  weight_decay: 0.01
  
  # Data configuration
  max_audio_length: 40000
  sample_rate: 16000
  mask_length: 10
  mask_prob: 0.08
  
  # Federated learning
  num_rounds: 2
  min_fit_clients: 2
  fraction_fit: 1.0
```

#### Distillation Configuration (`configs/distillation_config.yaml`)
```yaml
distillation:
  # Teacher model configuration
  teacher_model_path: "path/to/teacher/checkpoint"
  teacher_frozen: true
  
  # Distillation parameters
  alpha: 0.5  # Weight for distillation loss
  temperature: 2.0  # Softmax temperature
  
  # Student model configuration
  student_hidden_size: 384  # Smaller than teacher
  student_num_layers: 6
```

## 🚀 Execution Flow

### 1. Data Preparation Phase
```bash
# 1. Partition LibriSpeech dataset
python partition_data.py

# 2. Generate K-means targets for each client
python generate_kmeans_targets.py
```

### 2. Training Phase
```bash
# 3. Run federated pretraining
./run_pretraining.sh

# 4. Run knowledge distillation (optional)
./run_distillation.sh
```

### 3. Evaluation Phase
```bash
# 5. Fine-tune on downstream tasks
python s3prl/run_downstream.py -m train -u hubert_local \
    -k checkpoints/pretraining/server/best_global_model.pt \
    -d speech_commands
```

## 🔍 Memory and Performance Optimization

### 1. Gradient Checkpointing
```python
# Enable in configuration
gradient_checkpointing: true

# Implementation in model
if self.gradient_checkpointing and self.training:
    return checkpoint(self._forward_impl, x, use_reentrant=False)
```

### 2. Mixed Precision Training
```python
# Automatic mixed precision
scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. Dynamic Batch Sizing
```python
def get_optimal_batch_size(available_memory: float, model_size: float) -> int:
    """Calculate optimal batch size based on available memory"""
    # Estimate memory per sample
    # Calculate safe batch size
    # Return optimal value
```

## 🧪 Testing and Validation

### 1. Unit Tests
```python
def test_hubert_model_forward():
    """Test HuBERT model forward pass"""
    model = HubertBase(hidden_size=128, num_layers=2)
    x = torch.randn(2, 1000, 128)  # [batch, seq_len, hidden]
    output = model(x)
    assert output.shape == (2, 1000, 128)
```

### 2. Integration Tests
```python
def test_federated_training_round():
    """Test complete federated training round"""
    # Setup clients and server
    # Execute training round
    # Verify parameter updates
    # Check loss convergence
```

### 3. Performance Benchmarks
```python
def benchmark_training_speed():
    """Benchmark training performance"""
    # Measure time per epoch
    # Monitor memory usage
    # Track GPU utilization
    # Compare with baseline
```

## 🐛 Debugging and Troubleshooting

### 1. Common Issues

#### Memory Errors
```python
# Solution: Enable gradient checkpointing
config['gradient_checkpointing'] = True

# Solution: Reduce batch size
config['batch_size'] = max(1, config['batch_size'] // 2)
```

#### Data Loading Issues
```python
# Check file paths
logger.info(f"Loading from: {audio_path}")
logger.info(f"File exists: {audio_path.exists()}")

# Verify audio format
logger.info(f"Audio shape: {audio.shape}")
logger.info(f"Sample rate: {sample_rate}")
```

#### Training Divergence
```python
# Monitor gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm > 10.0:
            logger.warning(f"Large gradient in {name}: {grad_norm}")
```

### 2. Logging and Monitoring

#### Structured Logging
```python
import logging
import json

def log_training_metrics(epoch: int, loss: float, metrics: Dict):
    """Log training metrics in structured format"""
    log_data = {
        'epoch': epoch,
        'loss': loss,
        'metrics': metrics,
        'timestamp': time.time()
    }
    logger.info(json.dumps(log_data))
```

#### Resource Monitoring
```python
def monitor_resources():
    """Monitor system resources during training"""
    import psutil
    
    memory = psutil.virtual_memory()
    gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    logger.info(f"CPU Memory: {memory.percent}%")
    logger.info(f"GPU Memory: {gpu_memory / 1024**3:.2f} GB")
```

## 📈 Performance Metrics

### 1. Training Metrics
- **Loss**: Pretraining loss (masked language modeling)
- **Accuracy**: Frame-level prediction accuracy
- **Perplexity**: Model uncertainty measure
- **Learning Rate**: Current learning rate value

### 2. Federated Metrics
- **Round Time**: Time per federated round
- **Communication Cost**: Parameter transfer size
- **Client Participation**: Active client count
- **Convergence**: Loss convergence across rounds

### 3. Resource Metrics
- **Memory Usage**: Peak memory consumption
- **GPU Utilization**: GPU usage percentage
- **Training Speed**: Samples per second
- **Efficiency**: FLOPS per second

## 🔮 Future Enhancements

### 1. Model Improvements
- **Architecture**: Larger models (HuBERT-Large)
- **Efficiency**: Quantization and pruning
- **Adaptation**: Domain adaptation techniques

### 2. Federated Learning
- **Privacy**: Differential privacy integration
- **Robustness**: Byzantine attack resistance
- **Scalability**: Hierarchical aggregation

### 3. Downstream Tasks
- **Speech Recognition**: ASR fine-tuning
- **Speaker Diarization**: Speaker identification
- **Emotion Recognition**: Sentiment analysis

## 📚 Additional Resources

- **Code Documentation**: Inline docstrings and type hints
- **Research Papers**: HuBERT and federated learning references
- **Community**: Flower framework documentation and examples
- **Benchmarks**: Performance comparison with other models
