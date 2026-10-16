import torch
import numpy 

def layer_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Perform Layer Normalization.
    """
    x = torch.tensor(X)
    gamma = torch.tensor(gamma)
    beta = torch.tensor(beta)
    eps = torch.tensor(epsilon)

    mean = torch.mean(x, dim = -1, keepdim = True)
    std = torch.sqrt(torch.sum((x - mean)**2 / x.shape[2], dim = -1, keepdim = True) + eps)

    return (x - mean) / std * gamma + beta