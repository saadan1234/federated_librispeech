#!/usr/bin/env python3
"""
Speech Commands finetuning script using S3PRL framework with federated pretrained HuBERT
"""

import argparse
import os
import sys
import torch
import yaml
from pathlib import Path

# Add S3PRL to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "s3prl"))

def main():
    parser = argparse.ArgumentParser(description="Finetune HuBERT on Speech Commands using S3PRL")
    
    # Model arguments
    parser.add_argument("--upstream", type=str, default="custom_hubert_local", 
                       help="Upstream model name")
    parser.add_argument("--ckpt", type=str, 
                       default="/home/saadan/scratch/federated_librispeech/src/checkpoints/pretraining/server/best_global_model.pt",
                       help="Path to pretrained HuBERT checkpoint")
    parser.add_argument("--downstream", type=str, default="speech_commands",
                       help="Downstream task name")
    parser.add_argument("--config", type=str, 
                       default="/home/saadan/scratch/federated_librispeech/src/configs/speech_commands_config.yaml",
                       help="Path to downstream task config")
    
    # Training arguments
    parser.add_argument("--expdir", type=str, default="/home/saadan/scratch/federated_librispeech/src/exp/speech_commands",
                       help="Experiment directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--njobs", type=int, default=8,
                       help="Number of parallel jobs")
    
    # Evaluation arguments
    parser.add_argument("--eval", action="store_true",
                       help="Run evaluation only")
    parser.add_argument("--test", action="store_true",
                       help="Run test evaluation")
    
    # S3PRL specific arguments
    parser.add_argument("--override", type=str, default="",
                       help="Override config parameters")
    parser.add_argument("--upstream_feature_selection", type=str, default="last_hidden_state",
                       help="Which upstream features to use")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
        # Memory optimization settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Enable memory efficient attention
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        
        # Set memory management options for better performance
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.8'
        
        # Enable CUDA launch blocking for debugging if needed
        if os.getenv('DEBUG_CUDA', '0') == '1':
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Create experiment directory
    os.makedirs(args.expdir, exist_ok=True)
    
    # Import S3PRL runner
    from s3prl.run_downstream import main as s3prl_main
    
    # Prepare S3PRL arguments
    s3prl_args = [
        "--mode", "train" if not args.eval else "evaluate",
        "--upstream", args.upstream,
        "--upstream_ckpt", args.ckpt,
        "--downstream", args.downstream,
        "--config", args.config,
        "--expdir", args.expdir,
        "--seed", str(args.seed),
        "--device", device,
        "--upstream_feature_selection", args.upstream_feature_selection
    ]
    
    # Add override parameters
    if args.override:
        s3prl_args.extend(["--override", args.override])
    
    # Add test flag if specified
    if args.test:
        s3prl_args.append("--test")
    
    print("Running S3PRL with arguments:", s3prl_args)
    
    # Set sys.argv for S3PRL
    original_argv = sys.argv
    sys.argv = ["s3prl"] + s3prl_args
    
    try:
        # Run S3PRL
        s3prl_main()
    except SystemExit:
        # S3PRL calls sys.exit(), catch it
        pass
    finally:
        # Restore original sys.argv
        sys.argv = original_argv
    
    print(f"Training completed! Results saved to {args.expdir}")

if __name__ == "__main__":
    main()