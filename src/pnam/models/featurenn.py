from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeature(torch.nn.Module):
    """Gaussian Fourier feature mapping."""

    def __init__(self, std: float, num_inputs: int, num_outputs: int) -> None:
        super(FourierFeature, self).__init__()
        gamma = 2. * torch.pi * torch.normal(0., std, (num_inputs, num_outputs))
        self.gamma = torch.nn.Parameter(gamma, requires_grad=False)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs @ self.gamma
        return torch.cat((torch.cos(outputs), torch.sin(outputs)), dim=-1)


class FeatureNN(torch.nn.Module):
    """Neural network model for each individual feature."""

    def __init__(
        self,
        std: float = 0.,
        num_inputs: int = 1,
        hidden_sizes: Sequence[int] = (64, 32),
        num_outputs: int = 1,
        activation: Callable = F.silu,
        dropout: float = 0.
    ) -> None:
        """Initialize `FeatureNN` hyperparameters.

        Args:
            std: Standard deviation of Gaussian Fourier feature mapping.
            num_inputs: Dimensionality of input data.
            hidden_sizes: Hidden dimensions for each layer.
            num_outputs: Dimensionality of output data.
            activation: Activation function for entire model.
            dropout: Coefficient for dropout regularization.
        """
        super(FeatureNN, self).__init__()
        self.std = std
        self.num_inputs = num_inputs
        self.hidden_sizes = hidden_sizes
        self.num_outputs = num_outputs
        self.activation = activation
        
        self.dropout = nn.Dropout(p=dropout)

        layers = []

        # First layer
        if self.std > 0:
            self.fourier = FourierFeature(
                std=self.std,
                num_inputs=self.num_inputs,
                num_outputs=int(self.hidden_sizes[0] / 2)
            )
        else:
            layers.append(nn.Linear(
                in_features=self.num_inputs, out_features=self.hidden_sizes[0]
            ))

        # Hidden layers
        for in_features, out_features in zip(
            self.hidden_sizes, self.hidden_sizes[1:]
        ):
            layers.append(nn.Linear(in_features, out_features))

        # Last `Linear` layer
        # Omit bias for single-input bases, made redundant by model bias
        layers.append(nn.Linear(
            in_features=self.hidden_sizes[-1],
            out_features=self.num_outputs,
            bias=False if self.num_inputs == 1 else True
        ))

        self.model = nn.ModuleList(layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute `FeatureNN` outputs."""
        outputs = inputs
        if self.num_inputs == 1:
            outputs = outputs.unsqueeze(1)
        if self.std > 0:
            outputs = self.fourier(outputs)
        for layer in self.model[:-1]:
            outputs = self.dropout(self.activation(layer(outputs)))
        outputs = self.model[-1](outputs)
        return outputs
