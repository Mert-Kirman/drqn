import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Locks down all sources of randomness for reproducible results.
    """
    # Set Python's built-in random module
    random.seed(seed)
    
    # Set environment variable for Python hash seed (for dict/set ordering)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set PyTorch's random seed
    torch.manual_seed(seed)
    
    # Set CUDA/GPU random seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Force cuDNN to be deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Global seed set to: {seed}")