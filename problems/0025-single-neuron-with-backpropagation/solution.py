import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    # Your code here
    initial_weights = torch.tensor(initial_weights).unsqueeze(dim=0)
    initial_weights = initial_weights.requires_grad_(True)

    initial_bias = torch.tensor(initial_bias)
    initial_bias = initial_bias.requires_grad_(True)

    labels = torch.tensor(labels).unsqueeze(dim=1).float()

    features = torch.tensor(features)
    mse_list = []

    optimizer = torch.optim.SGD([initial_weights,initial_bias],learning_rate)
    for epoch in range(epochs):

        logits = torch.einsum('nx,kx->nk',features,initial_weights)
        logits += initial_bias
        output = torch.sigmoid(logits)
        loss = torch.nn.functional.mse_loss(output,labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        mse_list.append(round(loss.item(),4))

    return (initial_weights.tolist(),initial_bias.tolist(),mse_list)
    