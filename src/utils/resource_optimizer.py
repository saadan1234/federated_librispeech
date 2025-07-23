#!/usr/bin/env python3
"""
Dynamic Resource Detection and Optimization for Federated HuBERT Pretraining
Automatically detects and configures optimal resource allocation
"""

import os
import psutil
import torch
import logging
import yaml
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class ResourceOptimizer:
    """Dynamically detect and optimize system resources for federated learning"""
    
    def __init__(self):
        self.num_cpus = self._detect_cpu_cores()
        self.total_memory_gb = self._detect_memory()
        self.gpu_info = self._detect_gpus()
        self.optimal_workers = self._calculate_optimal_workers()
        
    def _detect_cpu_cores(self) -> int:
        """Detect total number of CPU cores"""
        try:
            # Use multiple methods to get accurate CPU count
            logical_cpus = psutil.cpu_count(logical=True)
            physical_cpus = psutil.cpu_count(logical=False)
            
            # Try to get more accurate count from system
            try:
                # Check for SLURM allocation
                if 'SLURM_CPUS_PER_TASK' in os.environ:
                    slurm_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
                    logger.info(f"SLURM CPU allocation detected: {slurm_cpus} CPUs")
                    return slurm_cpus
                elif 'SLURM_CPUS_ON_NODE' in os.environ:
                    slurm_cpus = int(os.environ['SLURM_CPUS_ON_NODE'])
                    logger.info(f"SLURM node CPUs detected: {slurm_cpus} CPUs")
                    return slurm_cpus
            except (ValueError, KeyError):
                pass
            
            # Fall back to psutil
            logger.info(f"CPU detection: {logical_cpus} logical, {physical_cpus} physical cores")
            return logical_cpus
            
        except Exception as e:
            logger.warning(f"Error detecting CPUs: {e}, defaulting to 1")
            return 1
    
    def _detect_memory(self) -> float:
        """Detect total available memory in GB"""
        try:
            # Check for SLURM memory allocation
            if 'SLURM_MEM_PER_NODE' in os.environ:
                slurm_mem_mb = int(os.environ['SLURM_MEM_PER_NODE'])
                slurm_mem_gb = slurm_mem_mb / 1024
                logger.info(f"SLURM memory allocation detected: {slurm_mem_gb:.1f}GB")
                return slurm_mem_gb
            elif 'SLURM_MEM_PER_CPU' in os.environ:
                slurm_mem_per_cpu_mb = int(os.environ['SLURM_MEM_PER_CPU'])
                total_mem_gb = (slurm_mem_per_cpu_mb * self.num_cpus) / 1024
                logger.info(f"SLURM per-CPU memory detected: {total_mem_gb:.1f}GB total")
                return total_mem_gb
            
            # Fall back to psutil
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            
            logger.info(f"Memory detection: {total_gb:.1f}GB total, {available_gb:.1f}GB available")
            return available_gb
            
        except Exception as e:
            logger.warning(f"Error detecting memory: {e}, defaulting to 8GB")
            return 8.0
    
    def _detect_gpus(self) -> Dict[str, Any]:
        """Detect available GPUs and their properties"""
        gpu_info = {
            'available': False,
            'count': 0,
            'total_memory_gb': 0,
            'devices': []
        }
        
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_info['available'] = True
                gpu_info['count'] = gpu_count
                
                total_memory = 0
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / (1024**3)
                    total_memory += memory_gb
                    
                    gpu_info['devices'].append({
                        'id': i,
                        'name': props.name,
                        'memory_gb': memory_gb,
                        'compute_capability': f"{props.major}.{props.minor}"
                    })
                    
                    logger.info(f"GPU {i}: {props.name} ({memory_gb:.1f}GB)")
                
                gpu_info['total_memory_gb'] = total_memory
                logger.info(f"Total GPU memory: {total_memory:.1f}GB across {gpu_count} devices")
            else:
                logger.info("No GPUs detected or CUDA not available")
                
        except Exception as e:
            logger.warning(f"Error detecting GPUs: {e}")
        
        return gpu_info
    
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of data loader workers"""
        # Use 75% of CPU cores for data loading, but cap at reasonable limits
        optimal = max(1, min(int(self.num_cpus * 0.75), 16))
        logger.info(f"Optimal data loader workers: {optimal}")
        return optimal
    
    def calculate_client_resources(self, num_clients: int) -> Dict[str, Any]:
        """Calculate optimal resource allocation per client"""
        
        # CPU allocation per client
        # Reserve some CPUs for system overhead
        available_cpus = max(1, self.num_cpus - 2)
        cpus_per_client = available_cpus / num_clients
        
        # Memory allocation per client (reserve 2GB for system)
        available_memory_gb = max(4, self.total_memory_gb - 2)
        memory_per_client_gb = available_memory_gb / num_clients
        memory_per_client_bytes = int(memory_per_client_gb * 1024**3)
        
        # GPU allocation per client
        gpu_per_client = 0.0
        if self.gpu_info['available'] and self.gpu_info['count'] > 0:
            # Distribute GPUs among clients
            gpu_per_client = min(1.0, self.gpu_info['count'] / num_clients)
        
        # Adjust batch size based on available GPU memory
        batch_size = self._calculate_optimal_batch_size()
        
        client_resources = {
            'num_cpus': round(cpus_per_client, 2),
            'num_gpus': round(gpu_per_client, 2),
            'memory': memory_per_client_bytes,
            'memory_gb': round(memory_per_client_gb, 1),
            'batch_size': batch_size,
            'num_workers': min(int(cpus_per_client), self.optimal_workers)
        }
        
        logger.info(f"Client resource allocation for {num_clients} clients:")
        logger.info(f"  CPUs per client: {client_resources['num_cpus']}")
        logger.info(f"  GPUs per client: {client_resources['num_gpus']}")
        logger.info(f"  Memory per client: {client_resources['memory_gb']}GB")
        logger.info(f"  Batch size: {client_resources['batch_size']}")
        logger.info(f"  Data workers: {client_resources['num_workers']}")
        
        return client_resources
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available GPU memory"""
        if not self.gpu_info['available']:
            return 2  # Small batch size for CPU-only training
        
        # Estimate memory usage per sample (rough approximation for HuBERT)
        # HuBERT with 10s audio @ 16kHz ≈ 160k samples
        # With feature extraction and model forward pass ≈ 500MB per sample
        memory_per_sample_mb = 500
        
        # Use 70% of GPU memory for model, leave 30% for overhead
        avg_gpu_memory_gb = self.gpu_info['total_memory_gb'] / max(1, self.gpu_info['count'])
        usable_memory_mb = avg_gpu_memory_gb * 1024 * 0.7
        
        optimal_batch_size = max(1, int(usable_memory_mb / memory_per_sample_mb))
        
        # Cap batch size to reasonable limits
        optimal_batch_size = min(optimal_batch_size, 16)
        
        logger.info(f"Calculated optimal batch size: {optimal_batch_size} "
                   f"(based on {avg_gpu_memory_gb:.1f}GB GPU memory)")
        
        return optimal_batch_size
    
    def optimize_config(self, config_path: str, num_clients: int = None) -> Dict[str, Any]:
        """Optimize configuration based on detected resources"""
        
        # Load existing config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Use provided num_clients or extract from config
        if num_clients is None:
            num_clients = config['simulation']['num_supernodes']
        
        # Calculate optimal client resources
        client_resources = self.calculate_client_resources(num_clients)
        
        # Update simulation backend config
        config['simulation']['backend']['config']['client_resources'] = {
            'num_cpus': client_resources['num_cpus'],
            'num_gpus': client_resources['num_gpus'],
            'memory': client_resources['memory']
        }
        
        # Update Ray init args for better performance
        config['simulation']['backend']['config']['init_args'].update({
            'num_cpus': self.num_cpus,
            'num_gpus': self.gpu_info['count'] if self.gpu_info['available'] else None,
            'object_store_memory': int(self.total_memory_gb * 0.3 * 1024**3),  # 30% for object store
        })
        
        # Update batch size based on config structure
        if 'pretraining' in config:
            config['pretraining']['batch_size'] = client_resources['batch_size']
        elif 'distillation' in config:
            config['distillation']['batch_size'] = client_resources['batch_size']
        
        if 'client' in config and 'local_config' in config['client']:
            config['client']['local_config']['batch_size'] = client_resources['batch_size']
        
        # Update data loader workers
        config['data']['dataloader']['num_workers'] = client_resources['num_workers']
        
        # Adjust local epochs based on batch size (smaller batches = more epochs)
        local_epochs = 1
        if client_resources['batch_size'] <= 2:
            local_epochs = 3
        elif client_resources['batch_size'] <= 4:
            local_epochs = 2
        
        if 'pretraining' in config:
            config['pretraining']['local_epochs'] = local_epochs
        elif 'distillation' in config:
            config['distillation']['local_epochs'] = local_epochs
        
        # Adjust timeout based on expected training time
        base_timeout = 600  # 10 minutes
        timeout_multiplier = max(1, 8 / client_resources['batch_size'])  # More time for smaller batches
        config['server']['round_timeout'] = int(base_timeout * timeout_multiplier)
        
        logger.info(f"Configuration optimized for current hardware:")
        logger.info(f"  Updated batch size: {client_resources['batch_size']}")
        logger.info(f"  Updated local epochs: {local_epochs}")
        logger.info(f"  Updated data workers: {client_resources['num_workers']}")
        logger.info(f"  Updated round timeout: {config['server']['round_timeout']}s")
        
        return config
    
    def save_optimized_config(self, config_path: str, output_path: str = None, num_clients: int = None):
        """Save optimized configuration to file"""
        if output_path is None:
            output_path = config_path.replace('.yaml', '_optimized.yaml')
        
        optimized_config = self.optimize_config(config_path, num_clients)
        
        with open(output_path, 'w') as f:
            yaml.safe_dump(optimized_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Optimized configuration saved to: {output_path}")
        return output_path
    
    def print_resource_summary(self):
        """Print a summary of detected resources"""
        print("=" * 80)
        print("SYSTEM RESOURCE DETECTION SUMMARY")
        print("=" * 80)
        print(f"CPU Cores: {self.num_cpus}")
        print(f"Total Memory: {self.total_memory_gb:.1f}GB")
        print(f"Optimal Data Workers: {self.optimal_workers}")
        
        if self.gpu_info['available']:
            print(f"GPUs Available: {self.gpu_info['count']}")
            print(f"Total GPU Memory: {self.gpu_info['total_memory_gb']:.1f}GB")
            for gpu in self.gpu_info['devices']:
                print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
        else:
            print("GPUs: Not available")
        print("=" * 80)

def main():
    """CLI for resource optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize federated learning config for current hardware")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--output", type=str, help="Output path for optimized config")
    parser.add_argument("--num-clients", type=int, help="Number of clients to optimize for")
    parser.add_argument("--summary", action="store_true", help="Print resource summary only")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Create optimizer
    optimizer = ResourceOptimizer()
    
    if args.summary:
        optimizer.print_resource_summary()
    else:
        optimizer.print_resource_summary()
        optimized_path = optimizer.save_optimized_config(
            args.config, 
            args.output, 
            args.num_clients
        )
        print(f"\nOptimized configuration saved to: {optimized_path}")

if __name__ == "__main__":
    main()