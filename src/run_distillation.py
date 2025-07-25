#!/usr/bin/env python3
"""
Script to run federated HuBERT knowledge distillation experiment
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Federated HuBERT Knowledge Distillation")
    parser.add_argument("--config", type=str, default="configs/distillation_config.yaml", 
                       help="Path to distillation configuration file")
    parser.add_argument("--num-clients", type=int, default=10, 
                       help="Number of clients for simulation")
    parser.add_argument("--num-rounds", type=int, default=20, 
                       help="Number of training rounds")
    parser.add_argument("--no-optimize", action="store_true", 
                       help="Skip automatic resource optimization")
    
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    # Path("models").mkdir(exist_ok=True)

    
    # Build command
    cmd = [
        sys.executable, 
        "federated_hubert_distillation.py",
        "--config", args.config,
        "--simulation",
        "--num-clients", str(args.num_clients),
        "--num-rounds", str(args.num_rounds)
    ]
    
    if args.no_optimize:
        cmd.append("--no-optimize")
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the distillation experiment
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info("Distillation experiment completed successfully!")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Distillation experiment failed with return code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Distillation experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()