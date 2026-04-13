import torch
import torch.nn as nn

def debug_model_dimensions(model, input_size=(1, 3, 224, 224)):
    """
    Debug function to find actual output dimensions from each level
    """
    model.eval()
    
    with torch.no_grad():
        x = torch.randn(*input_size)
        print(f"Input shape: {x.shape}")
        x = x.to('cuda')
        # Forward through patch embedding
        x = model.patch_embed(x)
        print(f"After patch_embed: {x.shape}")
        
        level_features = []
        level_dims = []
        
        # Forward through each level and collect dimensions
        for i, level in enumerate(model.levels):
            x = level(x)
            level_features.append(x)
            level_dims.append(x.shape[1])  # Channel dimension
            print(f"Level {i} output shape: {x.shape} -> {x.shape[1]} channels")
        
        print(f"\nActual level dimensions: {level_dims}")
        return level_dims, level_features