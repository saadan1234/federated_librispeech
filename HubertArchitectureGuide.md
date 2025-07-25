# HuBERT Federated Pretraining Architecture Guide

## Overview
This document explains the HuBERT (Hidden-Unit BERT) architecture implementation for federated self-supervised pretraining, designed for downstream tasks like speech command recognition.

## 🏗️ Model Architecture

### 1. Feature Encoder (Convolutional)
**Purpose:** Convert raw audio waveforms to high-level features

**Architecture:**
```
Input: (batch_size, audio_length) → (batch_size, 1, audio_length)
↓
7 Convolutional Layers:
- Layer 1: Conv1d(1→512, kernel=10, stride=5) + GroupNorm + GELU
- Layer 2: Conv1d(512→512, kernel=3, stride=2) + GroupNorm + GELU
- Layer 3: Conv1d(512→512, kernel=3, stride=2) + GroupNorm + GELU
- Layer 4: Conv1d(512→512, kernel=3, stride=2) + GroupNorm + GELU
- Layer 5: Conv1d(512→512, kernel=3, stride=2) + GroupNorm + GELU
- Layer 6: Conv1d(512→512, kernel=2, stride=2) + GroupNorm + GELU
- Layer 7: Conv1d(512→512, kernel=2, stride=2) + GroupNorm + GELU
↓
Projection: Linear(512→768) + Dropout
Output: (batch_size, seq_len, 768)
```

**Sequence Reduction:** Total stride = 5 × 2⁶ = 320
- Input: 160,000 samples → Output: 500 features

### 2. Transformer Encoder
**Purpose:** Learn contextual representations through self-attention

**Architecture:**
```
12 Transformer Layers, each containing:
- Multi-Head Self-Attention (12 heads, 768 dim)
- Layer Normalization
- Feed-Forward Network (768→3072→768)
- Layer Normalization
- Residual Connections
```

**Position Embeddings:** Learned positional encodings added to features

### 3. Prediction Head
**Purpose:** Predict discrete targets for masked positions

**Architecture:**
```
Linear(768 → vocab_size)  # vocab_size = 504 (k-means clusters)
```

## 🎯 Learning Process (Step-by-Step)

### Phase 1: Feature Extraction
1. **Raw Audio Input:** 16kHz audio samples (max 160,000 samples)
2. **Convolutional Processing:** 7-layer CNN extracts acoustic features
3. **Feature Output:** 768-dimensional features at 50Hz (320x downsampling)

### Phase 2: Span Masking
1. **Mask Generation:** Non-overlapping spans of 10 frames masked with 8% probability
2. **Mask Token:** Learnable mask token replaces masked features
3. **Example:** If sequence length is 500, ~40 frames are masked in 4 non-overlapping spans

### Phase 3: Contextual Learning
1. **Transformer Processing:** 12-layer transformer processes masked features
2. **Self-Attention:** Model learns relationships between masked and unmasked positions
3. **Contextual Representations:** Each position learns from entire sequence context

### Phase 4: Target Prediction
1. **K-means Targets:** Pre-computed cluster assignments from offline clustering
2. **Loss Computation:** Cross-entropy loss only on masked positions
3. **Learning Objective:** Predict correct cluster for each masked position

## 🔄 Federated Learning Setup

### Data Partitioning
```
Root Data Directory/
├── client_0/
│   ├── manifest.csv
│   ├── audio_files/
│   └── kmeans_targets.npy (optional)
├── client_1/
│   ├── manifest.csv
│   ├── audio_files/
│   └── kmeans_targets.npy (optional)
└── ...
```

### Training Process
1. **Client Initialization:** Each client loads its partitioned data
2. **Local Training:** Clients train on local data for multiple epochs
3. **Parameter Aggregation:** Server aggregates model parameters using FedAdam
4. **Global Update:** Updated model distributed to all clients
5. **Repeat:** Process continues for specified number of rounds

### Checkpoint Management
- **Client Checkpoints:** Saved as `state_dict` for s3prl compatibility
- **Server Checkpoints:** Global model saved after each round
- **Resume Capability:** Training can resume from any checkpoint

## 🎯 Downstream Task: Speech Command Recognition

### Fine-tuning Process
1. **Load Pretrained Model:** Use federated pretrained HuBERT
2. **Add Classification Head:** Linear layer on top of transformer outputs
3. **Task-Specific Training:** Fine-tune on speech command dataset
4. **Evaluation:** Test on held-out speech command data

### Expected Benefits
- **Better Representations:** Self-supervised pretraining learns robust audio features
- **Federated Privacy:** Training without sharing raw audio data
- **Transfer Learning:** Pretrained features transfer well to new tasks

## ⚙️ Configuration

### Key Parameters
```yaml
pretraining:
  vocab_size: 504              # Number of k-means clusters
  hidden_size: 768             # Transformer hidden dimension
  num_hidden_layers: 12        # Number of transformer layers
  num_attention_heads: 12      # Number of attention heads
  intermediate_size: 3072      # Feed-forward hidden size
  mask_prob: 0.08              # Masking probability
  mask_length: 10              # Span length for masking
  kmeans_targets_file: "kmeans_targets.npy"  # Optional: path to pre-computed targets
  auto_generate_kmeans: true   # Auto-generate targets if file not found
```

### Training Parameters
```yaml
pretraining:
  batch_size: 8
  learning_rate: 1e-4
  local_epochs: 3
  num_rounds: 100
  max_audio_length: 160000
```

## 🚀 Usage Instructions

### 1. Prepare Data
```bash
# Partition your LibriSpeech data into client directories
# Each client should have: manifest.csv, audio files
# K-means targets will be auto-generated if not provided
```

### 2. K-means Targets (Three Options)

#### Option A: Auto-Generation (Recommended for First Run)
```yaml
# In your config file:
pretraining:
  auto_generate_kmeans: true    # Will auto-generate if no file found
  kmeans_targets_file: null     # No pre-computed file needed
```

#### Option B: Pre-computed Targets
```python
# Generate targets manually (if you want more control)
from federated_hubert_pretraining import generate_kmeans_targets

# Extract features from your audio data
features = extract_features(audio_data)  # Shape: (num_samples, seq_len, 768)

# Generate k-means targets
targets = generate_kmeans_targets(features, n_clusters=504)
np.save("kmeans_targets.npy", targets)

# Then in config:
pretraining:
  kmeans_targets_file: "kmeans_targets.npy"
  auto_generate_kmeans: false
```

#### Option C: No Targets (Not Recommended)
```yaml
# Training without k-means targets (will skip loss computation)
pretraining:
  auto_generate_kmeans: false
  kmeans_targets_file: null
```

### 3. Run Federated Training
```bash
python src/federated_hubert_pretraining.py --simulation --num-clients 4 --num-rounds 50
```

### 4. Fine-tune for Speech Commands
```python
# Load pretrained model
model = HuBERTPretrainingModel.from_pretrained("checkpoints/best_global_model_state_dict.pt")

# Add classification head
classifier = nn.Linear(768, num_speech_commands)

# Fine-tune on speech command dataset
# ... fine-tuning code ...
```

## 📊 Expected Performance

### Pretraining Metrics
- **Loss:** Should decrease from ~6.0 to ~2.0-3.0
- **Convergence:** Typically 50-100 federated rounds
- **Memory:** ~8GB GPU memory for batch_size=8

### Downstream Performance
- **Speech Commands:** Expected 90%+ accuracy with proper fine-tuning
- **Transfer Learning:** Better than training from scratch
- **Robustness:** More robust to noise and variations

## 🔧 Troubleshooting

### Common Issues
1. **Memory Errors:** Reduce batch_size or max_audio_length
2. **Slow Training:** Reduce local_epochs or use fewer clients
3. **Poor Convergence:** Check k-means target quality or adjust learning rate
4. **Data Loading Errors:** Verify manifest.csv format and file paths

### Debugging Tips
- Check logs in `logs/federated_pretraining.log`
- Monitor loss curves during training
- Validate k-means targets alignment with feature sequence length
- Ensure proper data partitioning across clients

## 📚 References

1. **HuBERT Paper:** "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units"
2. **Federated Learning:** Flower framework documentation
3. **Speech Commands:** Google Speech Commands dataset
4. **s3prl:** Self-Supervised Speech Pre-training and Representation Learning Toolkit 