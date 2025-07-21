#!/usr/bin/env python3

import torch
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from federated_hubert_pretraining import HuBERTPretrainingModel
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

def test_parameter_consistency():
    """Test that server and client models have consistent parameter structures."""
    
    print("Testing parameter consistency between server and client models...")
    
    # Model configuration from the config file
    model_config = {
        'vocab_size': 504,
        'hidden_size': 768,
        'num_hidden_layers': 12,
        'num_attention_heads': 12,
        'intermediate_size': 3072,
        'mask_prob': 0.08,
        'mask_length': 10
    }
    
    # Create server model (fresh initialization)
    print("1. Creating server model...")
    server_model = HuBERTPretrainingModel(**model_config)
    server_params = [param.cpu().numpy() for param in server_model.state_dict().values()]
    server_param_names = list(server_model.state_dict().keys())
    
    print(f"   Server model has {len(server_params)} parameters")
    print(f"   Parameter names: {server_param_names[:5]}..." if len(server_param_names) > 5 else f"   Parameter names: {server_param_names}")
    print(f"   Parameter shapes: {[p.shape for p in server_params[:5]]}" if len(server_params) > 5 else f"   Parameter shapes: {[p.shape for p in server_params]}")
    
    # Create client model
    print("\n2. Creating client model...")
    client_model = HuBERTPretrainingModel(**model_config)
    
    # Simulate client model after a forward pass (this is where the issue might occur)
    print("3. Simulating forward pass on client model...")
    dummy_input = torch.randn(2, 160000)  # batch_size=2, max_length=160000
    dummy_attention_mask = torch.ones(2, 160000)
    
    with torch.no_grad():
        outputs = client_model(dummy_input, dummy_attention_mask)
    
    client_params = [param.cpu().numpy() for param in client_model.state_dict().values()]
    client_param_names = list(client_model.state_dict().keys())
    
    print(f"   Client model has {len(client_params)} parameters after forward pass")
    print(f"   Parameter names: {client_param_names[:5]}..." if len(client_param_names) > 5 else f"   Parameter names: {client_param_names}")
    print(f"   Parameter shapes: {[p.shape for p in client_params[:5]]}" if len(client_params) > 5 else f"   Parameter shapes: {[p.shape for p in client_params]}")
    
    # Test parameter consistency
    print("\n4. Testing parameter consistency...")
    
    # Check number of parameters
    if len(server_params) != len(client_params):
        print(f"   ❌ FAIL: Parameter count mismatch - Server: {len(server_params)}, Client: {len(client_params)}")
        return False
    else:
        print(f"   ✅ PASS: Parameter count matches ({len(server_params)})")
    
    # Check parameter names
    if server_param_names != client_param_names:
        print("   ❌ FAIL: Parameter names mismatch")
        print(f"      Server names: {server_param_names}")
        print(f"      Client names: {client_param_names}")
        return False
    else:
        print("   ✅ PASS: Parameter names match")
    
    # Check parameter shapes
    shape_mismatch = False
    for i, (server_shape, client_shape) in enumerate(zip([p.shape for p in server_params], [p.shape for p in client_params])):
        if server_shape != client_shape:
            print(f"   ❌ FAIL: Shape mismatch at parameter {i} ({server_param_names[i]}): Server {server_shape} vs Client {client_shape}")
            shape_mismatch = True
    
    if not shape_mismatch:
        print("   ✅ PASS: All parameter shapes match")
    
    # Test Flower parameter conversion
    print("\n5. Testing Flower parameter conversion...")
    try:
        # Convert to Flower Parameters format
        server_params_flower = ndarrays_to_parameters(server_params)
        client_params_flower = ndarrays_to_parameters(client_params)
        
        # Convert back to check consistency
        server_params_back = parameters_to_ndarrays(server_params_flower)
        client_params_back = parameters_to_ndarrays(client_params_flower)
        
        print("   ✅ PASS: Flower parameter conversion works")
        
        # Test aggregation simulation (what happens during FedAdam)
        print("\n6. Testing parameter aggregation simulation...")
        
        # Simple averaging (similar to what happens in federated learning)
        try:
            aggregated_params = []
            for server_p, client_p in zip(server_params, client_params):
                # This is where the broadcast error would occur
                avg_param = (server_p + client_p) / 2
                aggregated_params.append(avg_param)
            
            print("   ✅ PASS: Parameter aggregation simulation successful")
            
        except Exception as e:
            print(f"   ❌ FAIL: Parameter aggregation failed: {e}")
            return False
            
    except Exception as e:
        print(f"   ❌ FAIL: Flower parameter conversion failed: {e}")
        return False
    
    if shape_mismatch:
        return False
    
    print("\n🎉 All tests passed! Parameter structures are consistent.")
    return True

if __name__ == "__main__":
    success = test_parameter_consistency()
    sys.exit(0 if success else 1)