class S3PRLCompatibleClient(NumPyClient):
    """
    Federated client that maintains full s3prl compatibility for research comparison.
    All training logic follows s3prl's HuBERT pretraining procedures exactly.
    """

    def __init__(self, client_id: int, train_dataset, val_dataset, model, device=None):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model

        # Device setup with proper GPU utilization for your allocation
        if device is None:
            if torch.cuda.is_available():
                # Use GPU since you have good allocation
                self.device = torch.device("cuda:0")
                logger.info(f"Client {client_id}: Using GPU cuda:0")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.model = self.model.to(self.device)

        # Load s3prl compatible configuration with reasonable settings for your resources
        config_path = "/home/saadan/scratch/federated_librispeech/src/configs/pretraining_config.yaml"
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            cfg = {'pretraining': {'batch_size': 2, 'num_workers': 2}}
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            cfg = {'pretraining': {'batch_size': 2, 'num_workers': 2}}

        self.config = cfg.get('pretraining', {})
        
        # Reasonable dataloader settings for your GPU allocation
        batch_size = min(self.config.get('batch_size', 2), 4)  # Conservative but reasonable
        num_workers = min(self.config.get('num_workers', 2), 3)  # Use some of your CPUs
        
        logger.info(f"Client {client_id}: Using batch_size={batch_size}, num_workers={num_workers}")
        
        # Create dataloaders with GPU-optimized settings
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,  # Enable for GPU
            collate_fn=collate_fn,
            drop_last=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,  # Enable for GPU
            collate_fn=collate_fn,
            drop_last=False
        )

        logger.info(f"Client {client_id}: Initialized with {len(self.train_dataset)} train, {len(self.val_dataset)} val samples")

// ...existing code...

def client_fn(context: Context) -> S3PRLCompatibleClient:
    """Create client function that uses the exact partitioned data structure."""
    
    # Add memory cleanup but allow GPU usage
    import gc
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    config_path = "/home/saadan/scratch/federated_librispeech/src/configs/pretraining_config.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = {
            'simulation': {'num_supernodes': 2},
            'pretraining': {
                'max_audio_length': 160000,  # Reduced from 250000 for memory efficiency
                'sample_rate': 16000, 'label_rate': 50,
                'extractor_mode': 'default', 
                'num_hidden_layers': 8,  # Reduced from 12 for memory
                'hidden_size': 512,  # Reduced from 768 for memory
                'intermediate_size': 2048,  # Reduced from 3072 for memory
                'num_attention_heads': 8,  # Reduced from 12 for memory
                'activation_fn': 'gelu',
                'dropout': 0.1, 'attention_dropout': 0.1, 'activation_dropout': 0.0,
                'final_dim': 256, 'layer_norm_first': True,
                'conv_feature_layers': "[(512,10,5)] + [(512,3,2)]*3 + [(512,2,2)]*1",  # Reduced layers
                'logit_temp': 0.1, 'mask_prob': 0.08, 'mask_selection': 'static',
                'mask_other': 0, 'mask_length': 10, 'no_mask_overlap': False,
                'mask_min_space': 1, 'conv_bias': False, 'encoder_layerdrop': 0.0,
                'dropout_input': 0.0, 'dropout_features': 0.0, 'feature_grad_mult': 0.1,
                'untie_final_proj': True, 'normalize': False, 'enable_padding': False,
                'max_keep_size': None, 'min_sample_size': None, 'random_crop': True,
                'pad_audio': False
            }
        }
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

    # Get client ID
    node_id = context.node_id
    num_clients = config.get('simulation', {}).get('num_supernodes', 2)
    client_id = hash(str(node_id)) % num_clients

    # Use the exact partitioned data structure
    base_path = Path("/home/saadan/scratch/federated_librispeech/src/federated_librispeech/data")
    client_path = base_path / f"client_{client_id}"

    if not client_path.exists():
        raise FileNotFoundError(f"Client data not found: {client_path}")

    # Load datasets using partitioned structure
    train_manifest = client_path / "train.csv"
    val_manifest = client_path / "validation.csv"
    kmeans_targets = client_path / "kmeans_targets.npy"

    if not all([train_manifest.exists(), val_manifest.exists(), kmeans_targets.exists()]):
        raise FileNotFoundError(f"Required files missing in {client_path}")

    # Load kmeans metadata for vocab size
    try:
        with open(base_path / "kmeans_metadata.json", 'r') as f:
            kmeans_meta = json.load(f)
        vocab_size = kmeans_meta.get('n_clusters', 504)
    except:
        vocab_size = 504

    pre_cfg = config.get('pretraining', {})

    # Load train and validation manifests to get sample counts
    train_df = pd.read_csv(train_manifest)
    val_df = pd.read_csv(val_manifest)
    
    # Limit dataset size to manageable amount for your GPU
    max_samples_per_split = 200  # Increased from 50 since you have good resources
    if len(train_df) > max_samples_per_split:
        train_df = train_df.iloc[:max_samples_per_split].copy()
        train_df.to_csv(train_manifest, index=False)
        logger.info(f"Client {client_id}: Limited train dataset to {max_samples_per_split} samples")
    
    if len(val_df) > max_samples_per_split // 4:  # Smaller validation set
        val_df = val_df.iloc[:max_samples_per_split // 4].copy()
        val_df.to_csv(val_manifest, index=False)
        logger.info(f"Client {client_id}: Limited validation dataset to {max_samples_per_split // 4} samples")
    
    logger.info(f"Client {client_id}: Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # ...existing target handling code...

    # Create datasets with the properly split targets
    train_dataset = FederatedLibriSpeechDataset(
        manifest_file=str(train_manifest),
        kmeans_targets_path=str(train_targets_path),
        split="train",
        max_length=pre_cfg.get('max_audio_length', 160000),  # Reduced for memory
        sample_rate=pre_cfg.get('sample_rate', 16000),
        label_rate=pre_cfg.get('label_rate', 50),
        vocab_size=vocab_size
    )

    val_dataset = FederatedLibriSpeechDataset(
        manifest_file=str(val_manifest),
        kmeans_targets_path=str(val_targets_path),
        split="validation",
        max_length=pre_cfg.get('max_audio_length', 160000),  # Reduced for memory
        sample_rate=pre_cfg.get('sample_rate', 16000),
        label_rate=pre_cfg.get('label_rate', 50),
        vocab_size=vocab_size
    )

    # Create s3prl HubertModel with reduced configuration optimized for your GPU
    model_cfg = HubertConfig(
        label_rate=pre_cfg.get('label_rate', 50),
        extractor_mode=pre_cfg.get('extractor_mode', "default"),
        encoder_layers=pre_cfg.get('num_hidden_layers', 8),  # Reduced
        encoder_embed_dim=pre_cfg.get('hidden_size', 512),  # Reduced
        encoder_ffn_embed_dim=pre_cfg.get('intermediate_size', 2048),  # Reduced
        encoder_attention_heads=pre_cfg.get('num_attention_heads', 8),  # Reduced
        activation_fn=pre_cfg.get('activation_fn', "gelu"),
        dropout=pre_cfg.get('dropout', 0.1),
        attention_dropout=pre_cfg.get('attention_dropout', 0.1),
        activation_dropout=pre_cfg.get('activation_dropout', 0.0),
        final_dim=pre_cfg.get('final_dim', 256),
        layer_norm_first=pre_cfg.get('layer_norm_first', True),
        conv_feature_layers=pre_cfg.get('conv_feature_layers', "[(512,10,5)] + [(512,3,2)]*3 + [(512,2,2)]*1"),
        logit_temp=pre_cfg.get('logit_temp', 0.1),
        mask_prob=pre_cfg.get('mask_prob', 0.08),
        mask_selection=pre_cfg.get('mask_selection', "static"),
        mask_other=pre_cfg.get('mask_other', 0),
        mask_length=pre_cfg.get('mask_length', 10),
        no_mask_overlap=pre_cfg.get('no_mask_overlap', False),
        mask_min_space=pre_cfg.get('mask_min_space', 1),
        conv_bias=pre_cfg.get('conv_bias', False),
        encoder_layerdrop=pre_cfg.get('encoder_layerdrop', 0.0),
        dropout_input=pre_cfg.get('dropout_input', 0.0),
        dropout_features=pre_cfg.get('dropout_features', 0.0),
        feature_grad_mult=pre_cfg.get('feature_grad_mult', 0.1),
        untie_final_proj=pre_cfg.get('untie_final_proj', True),
    )

    task_cfg = HubertPretrainingConfig(
        label_rate=pre_cfg.get('label_rate', 50),
        sample_rate=pre_cfg.get('sample_rate', 16000),
        normalize=pre_cfg.get('normalize', False),
        enable_padding=pre_cfg.get('enable_padding', False),
        max_keep_size=pre_cfg.get('max_keep_size', None),
        max_sample_size=pre_cfg.get('max_audio_length', 160000),  # Reduced
        min_sample_size=pre_cfg.get('min_sample_size', None),
        random_crop=pre_cfg.get('random_crop', True),
        pad_audio=pre_cfg.get('pad_audio', False)
    )

    # Create dictionaries for HuBERT model
    dictionaries = [list(range(vocab_size))]

    try:
        model = HubertModel(model_cfg, task_cfg, dictionaries)
        logger.info(f"Client {client_id}: Model created successfully with {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        logger.error(f"Client {client_id}: Failed to create model: {e}")
        raise

    return S3PRLCompatibleClient(
        client_id=client_id,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model
    )

// ...existing code...

def fit_config(server_round: int) -> Dict[str, Union[str, int, float]]:
    """Return training configuration dict for each round."""
    config = {
        "server_round": server_round,
        "local_epochs": 1,
        "lr": max(0.00005, 0.0002 * (0.98 ** (server_round - 1))),  # Adjusted learning rate
        "batch_size": 2,  # Reasonable for your GPU
    }
    return config

def evaluate_config(server_round: int) -> Dict[str, Union[str, int, float]]:
    """Return evaluation configuration dict for each round."""
    return {
        "server_round": server_round,
        "batch_size": 2,  # Reasonable for your GPU
    }

// ...existing code...

            try:
                # Use your GPU resources effectively
                history = run_simulation(
                    server_app=server_app,
                    client_app=client_app,
                    num_supernodes=args.num_clients,
                    backend_config={
                        "client_resources": {
                            "num_cpus": 2.0,  # Use some of your 8 CPUs
                            "num_gpus": 0.8   # Use most of your 1 GPU (leave some buffer)
                        }
                    }
                )