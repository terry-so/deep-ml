import torch

def entropy_and_cross_entropy(P: torch.Tensor, Q: torch.Tensor) -> tuple[float, float]:
    """
    Compute entropy of P and cross-entropy between P and Q.
    
    Args:
        P: True probability distribution (torch.Tensor)
        Q: Predicted probability distribution (torch.Tensor)
    
    Returns:
        Tuple of (entropy H(P), cross-entropy H(P,Q))
    """
    # Your code here
    P = torch.tensor(P)
    Q = torch.tensor(Q)
    entropy = - P@torch.log(P)
    entropy[torch.isnan(entropy)] = 0
    cross_entropy = -P@torch.log(Q)
    cross_entropy[torch.isnan(cross_entropy)] = 0

    return(entropy.tolist(),cross_entropy.tolist())