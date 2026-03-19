import torch

def select_layers(dataset, shots, cfg):
    """
    Placeholder algorithm to select layers for Multi-Modal Adapter.
    In the future, this will run on a small subset of the training data
    to dynamically identify the optimal layers.
    """
    print(f"Running CeKALA layer selection for {dataset} with {shots} shots...")
    # Future implementation: use a small subset of DataManager(cfg) to compute gradients
    # and find the most sensitive layers.
    
    # Hardcoded selected layers for now
    selected_layers = [6, 8, 10, 11]
    return selected_layers
