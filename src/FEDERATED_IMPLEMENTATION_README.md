# Federated Research-Standard HuBERT Implementation

This directory contains **federated learning implementations** of HuBERT pretraining and distillation that maintain **research-standard architecture** while enabling distributed training across multiple clients.

## 🎯 **Key Features**

✅ **Federated Learning**: Distributed training across multiple clients  
✅ **Research Standards**: Uses standard PyTorch components for benchmarking  
✅ **Checkpoint Compatibility**: Creates checkpoints compatible with `s3prl` and other frameworks  
✅ **Knowledge Distillation**: Teacher-student architecture with 4x compression  
✅ **Performance Optimized**: Mixed precision, gradient accumulation, optimized DataLoader  

## 🏗️ **Architecture Overview**

### **Federated Pretraining**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client 1      │    │   Client 2      │    │   Client N      │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │Research     │ │    │ │Research     │ │    │ │Research     │ │
│ │Standard     │ │    │ │Standard     │ │    │ │Standard     │ │
│ │HuBERT       │ │    │ │HuBERT       │ │    │ │HuBERT       │ │
│ │(768H, 12L)  │ │    │ │(768H, 12L)  │ │    │ │(768H, 12L)  │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Global Model  │
                    │   Aggregation   │
                    │   (FedAvg)      │
                    └─────────────────┘
```

### **Federated Distillation**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client 1      │    │   Client 2      │    │   Client N      │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │Teacher      │ │    │ │Teacher      │ │    │ │Teacher      │ │
│ │(768H, 12L)  │ │    │ │(768H, 12L)  │ │    │ │(768H, 12L)  │ │
│ │+ Student    │ │    │ │+ Student    │ │    │ │+ Student    │ │
│ │(384H, 6L)   │ │    │ │(384H, 6L)   │ │    │ │(384H, 6L)   │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Global Student  │
                    │   Aggregation   │
                    │   (FedAvg)      │
                    └─────────────────┘
```

## 🚀 **Quick Start**

### **1. Test the Implementation**
```bash
cd src
python3 test_federated_implementation.py
```

### **2. Run Federated Pretraining**
```bash
# With default settings
./run_research_standard_pretraining.sh

# Custom configuration
./run_research_standard_pretraining.sh \
    --config configs/research_standard_pretraining_config.yaml \
    --checkpoint_dir ./checkpoints/research_standard \
    --num_rounds 100
```

### **3. Run Federated Distillation**
```bash
# With default settings (uses your existing checkpoint)
./run_research_standard_distillation.sh

# Custom teacher checkpoint
./run_research_standard_distillation.sh \
    --teacher_checkpoint ./checkpoints/research_standard/round_010_checkpoint.pt \
    --checkpoint_dir ./checkpoints/research_standard_distillation
```

## 📊 **Federated Learning Parameters**

### **Client Configuration**
- **Number of clients**: Configurable (default: 10)
- **Clients per round**: Configurable fraction (default: 50%)
- **Data partitioning**: Random partitioning among clients
- **Train/Val split**: 80/20 per client

### **Aggregation Strategy**
- **Current**: Simple FedAvg (weighted by sample count)
- **Future**: FedAdam, FedProx, FedNova support
- **Communication**: Synchronous rounds

### **Training Flow**
1. **Round Start**: Select clients for this round
2. **Client Training**: Each client trains on local data
3. **Parameter Upload**: Clients send updated parameters
4. **Aggregation**: Server aggregates parameters (FedAvg)
5. **Global Update**: Update global model
6. **Evaluation**: Evaluate global model on validation set
7. **Checkpointing**: Save model if checkpoint interval reached

## 🔧 **Configuration Files**

### **Pretraining Configuration**
```yaml
federated:
  num_clients: 10           # Total number of federated clients
  num_rounds: 100           # Total federated rounds
  client_fraction: 0.5      # Fraction of clients per round
  
  # Aggregation strategy
  strategy: "FedAdam"       # Use FedAdam for better convergence
  beta1: 0.9               # FedAdam beta1
  beta2: 0.999             # FedAdam beta2
  eta: 0.01                # FedAdam learning rate
  tau: 0.001               # FedAdam tau
```

### **Distillation Configuration**
```yaml
federated:
  num_clients: 10           # Total number of federated clients
  num_rounds: 100           # Total federated rounds
  client_fraction: 0.5      # Fraction of clients per round
  
  # Aggregation strategy
  strategy: "FedAdam"       # Use FedAdam for better convergence
  beta1: 0.9               # FedAdam beta1
  beta2: 0.999             # FedAdam beta2
  eta: 0.01                # FedAdam learning rate
  tau: 0.001               # FedAdam tau
```

## 📁 **File Structure**

```
src/
├── research_standard_hubert_pretraining.py      # Federated pretraining
├── research_standard_hubert_distillation.py     # Federated distillation
├── configs/
│   ├── research_standard_pretraining_config.yaml    # Pretraining config
│   └── research_standard_distillation_config.yaml   # Distillation config
├── run_research_standard_pretraining.sh         # Pretraining script
├── run_research_standard_distillation.sh        # Distillation script
├── test_federated_implementation.py             # Test script
└── checkpoints/
    ├── research_standard/                        # Pretraining checkpoints
    └── research_standard_distillation/           # Distillation checkpoints
```

## 🎯 **Research Benefits**

### **Federated Learning Advantages**
- **Privacy**: Data stays on client devices
- **Scalability**: Train on larger distributed datasets
- **Efficiency**: Parallel training across multiple clients
- **Real-world**: Simulates real distributed scenarios

### **Research Standards Maintained**
- **Architecture**: Standard PyTorch `nn.TransformerEncoderLayer`
- **Parameters**: HuBERT paper specifications (768H, 12L, 504V)
- **Checkpoints**: Compatible with `s3prl`, HuggingFace, etc.
- **Benchmarking**: Comparable with centralized training results

### **Performance Metrics**
- **Convergence**: Track loss across federated rounds
- **Communication**: Monitor client participation and aggregation
- **Efficiency**: Measure training time and resource usage
- **Quality**: Compare with centralized training baselines

## 🔍 **Monitoring and Debugging**

### **Training Progress**
```bash
# Check checkpoint progress
ls -la checkpoints/research_standard/

# Monitor logs
tail -f logs/federated_training.log

# Check model parameters
python3 -c "
import torch
ckpt = torch.load('checkpoints/research_standard/round_010_checkpoint.pt', map_location='cpu')
print(f'Round: {ckpt[\"round\"]}')
print(f'Federated metrics: {ckpt.get(\"federated_metrics\", {})}')
"
```

### **Common Issues**
1. **Client failures**: Check client logs and data availability
2. **Aggregation errors**: Verify parameter shapes and data types
3. **Memory issues**: Reduce batch size or client fraction
4. **Communication**: Ensure network connectivity between clients

## 🚨 **Important Notes**

### **Data Requirements**
- **Manifest file**: CSV with audio paths and metadata
- **Audio data**: LibriSpeech format with proper structure
- **KMeans targets**: Precomputed pseudo-labels for HuBERT training
- **Data partitioning**: Automatic partitioning among clients

### **Hardware Requirements**
- **GPU**: CUDA-compatible GPU for training
- **Memory**: Sufficient RAM for model and data
- **Storage**: Space for checkpoints and logs
- **Network**: Stable connection for federated communication

### **Security Considerations**
- **Data privacy**: Data never leaves client devices
- **Parameter security**: Only model parameters are shared
- **Authentication**: Implement client authentication if needed
- **Encryption**: Consider parameter encryption for sensitive applications

## 🎉 **Expected Results**

### **Pretraining**
- **Federated convergence**: Loss reduction across rounds
- **Research compatibility**: Standard checkpoint format
- **Performance**: Comparable to centralized training
- **Scalability**: Handle large distributed datasets

### **Distillation**
- **Model compression**: 4x parameter reduction
- **Knowledge retention**: Maintain teacher performance
- **Federated efficiency**: Distributed compression training
- **Deployment ready**: Small, fast student models

## 🤝 **Next Steps**

1. **Test the implementation** with the test script
2. **Update configuration files** with your dataset paths
3. **Run small-scale training** to verify everything works
4. **Scale up** to full federated training
5. **Compare results** with centralized training baselines
6. **Publish research** with reproducible, benchmarkable results

**Your federated implementation maintains research standards while enabling distributed training!** 🚀
