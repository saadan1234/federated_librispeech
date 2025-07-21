#!/usr/bin/env python3
"""
Speech Commands evaluation script using S3PRL framework with federated pretrained HuBERT
"""

import argparse
import os
import sys
import torch
import json
from pathlib import Path

# Add S3PRL to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "s3prl"))

def main():
    parser = argparse.ArgumentParser(description="Evaluate HuBERT on Speech Commands using S3PRL")
    
    # Model arguments
    parser.add_argument("--upstream", type=str, default="custom_hubert_local", 
                       help="Upstream model name")
    parser.add_argument("--ckpt", type=str, 
                       default="/home/saadan/scratch/federated_librispeech/src/models/pretrained_hubert.pt",
                       help="Path to pretrained HuBERT checkpoint")
    parser.add_argument("--downstream", type=str, default="speech_commands",
                       help="Downstream task name")
    parser.add_argument("--config", type=str, 
                       default="/lustre07/scratch/saadan/federated_librispeech/speech_commands/speech_commands_config.yaml",
                       help="Path to downstream task config")
    
    # Evaluation arguments
    parser.add_argument("--expdir", type=str, default="./exp/speech_commands",
                       help="Experiment directory")
    parser.add_argument("--test", action="store_true",
                       help="Run test evaluation (default: dev)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Output arguments
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                       help="Output file for results")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
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
    
    # Check if experiment directory exists
    if not os.path.exists(args.expdir):
        print(f"Error: Experiment directory {args.expdir} does not exist!")
        print("Please run training first with finetune_speech_commands.py")
        sys.exit(1)
    
    # Import S3PRL runner
    from s3prl.run_downstream import main as s3prl_main
    
    # Prepare S3PRL arguments for evaluation
    s3prl_args = [
        "--mode", "evaluate",
        "--upstream", args.upstream,
        "--ckpt", args.ckpt,
        "--downstream", args.downstream,
        "--config", args.config,
        "--expdir", args.expdir,
        "--seed", str(args.seed),
        "--device", device
    ]
    
    # Add test flag if specified
    if args.test:
        s3prl_args.append("--test")
    
    if args.verbose:
        print("Running S3PRL evaluation with arguments:", s3prl_args)
    
    # Set sys.argv for S3PRL
    original_argv = sys.argv
    sys.argv = ["s3prl"] + s3prl_args
    
    try:
        # Run S3PRL evaluation
        s3prl_main()
    except SystemExit:
        # S3PRL calls sys.exit(), catch it
        pass
    finally:
        # Restore original sys.argv
        sys.argv = original_argv
    
    # Try to read and parse results
    results = {}
    
    # Look for results in common S3PRL output locations
    result_files = [
        os.path.join(args.expdir, "dev_best.json"),
        os.path.join(args.expdir, "test_best.json"),
        os.path.join(args.expdir, "result.json"),
        os.path.join(args.expdir, "dev_result.json"),
        os.path.join(args.expdir, "test_result.json")
    ]
    
    for result_file in result_files:
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                    results[os.path.basename(result_file)] = result_data
                    if args.verbose:
                        print(f"Found results in {result_file}")
            except Exception as e:
                if args.verbose:
                    print(f"Could not read {result_file}: {e}")
    
    # Save consolidated results
    output_path = os.path.join(args.expdir, args.output_file)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation completed! Results saved to {output_path}")
    
    # Print summary
    if results:
        print("\n=== EVALUATION SUMMARY ===")
        for file_name, data in results.items():
            print(f"\n{file_name}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
    else:
        print("No evaluation results found. Check the experiment directory for output files.")

if __name__ == "__main__":
    main()