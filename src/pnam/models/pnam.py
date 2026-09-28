from typing import Callable, Sequence, Tuple

import math
import torch
import torch.nn as nn

from pnam.models.featurenn import FeatureNN


class FC(torch.nn.Module):
    """Fully connected feedforward neural network."""

    def __init__(
        self,
        std: float,
        num_inputs: int,
        hidden_sizes: Sequence[int],
        num_outputs: int,
        activation: Callable,
        dropout: float
    ) -> None:
        super(FC, self).__init__()
        self.std = std
        self.num_inputs = num_inputs
        self.hidden_sizes = hidden_sizes
        self.num_outputs = num_outputs
        self.activation = activation
        self.dropout = dropout

        # Build `FeatureNN`
        self.feature_nns = FeatureNN(
            std=self.std,
            num_inputs=self.num_inputs,
            hidden_sizes=self.hidden_sizes,
            num_outputs=self.num_outputs,
            activation=self.activation,
            dropout=self.dropout
        )

    def calc_outputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return outputs computed by each feature network."""
        return self.feature_nns(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.calc_outputs(inputs)


class PNAM(torch.nn.Module):
    """Projected neural additive model."""

    def __init__(
        self,
        std: float,
        num_inputs: int,
        proj_size: int,
        hidden_sizes: Sequence[int],
        num_outputs: int,
        activation: Callable,
        dropout: float,
        feature_dropout: float
    ) -> None:
        super(PNAM, self).__init__()
        self.std = std
        self.num_inputs = num_inputs
        self.proj_size = proj_size
        self.hidden_sizes = hidden_sizes
        self.num_outputs = num_outputs
        self.activation = activation
        self.dropout = dropout
        self.feature_dropout = feature_dropout

        self.dropout_layer = nn.Dropout(p=self.feature_dropout)

        # Projection layer
        # For a vanilla neural additive model, set `proj_size` = 0
        self.num_networks = self.num_inputs
        if self.proj_size > 0:
            self.linear = nn.Linear(
                in_features=self.num_inputs,
                out_features=self.proj_size,
                bias=False
            )
            self.num_networks = self.proj_size

        # Build `FeatureNN`
        self.feature_nns = nn.ModuleList([FeatureNN(
            std=self.std,
            num_inputs=1,
            hidden_sizes=self.hidden_sizes,
            num_outputs=self.num_outputs,
            activation=self.activation,
            dropout=self.dropout
        ) for _ in range(self.num_networks)])

        self.weight = torch.nn.Parameter(torch.empty(
            (self.num_outputs, self.num_networks)
        ))
        self.bias = torch.nn.Parameter(torch.empty(self.num_outputs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # For details, see https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5.))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1. / math.sqrt(fan_in) if fan_in > 0 else 0.
        nn.init.uniform_(self.bias, -bound, bound)

    def calc_outputs(self, inputs: torch.Tensor) -> Tuple[
        Sequence[torch.Tensor], torch.Tensor
    ]:
        """Return outputs computed by each feature network."""
        proj_inputs = inputs
        if self.proj_size > 0:
            proj_inputs = self.linear(inputs)
        individual_outputs = [
            self.feature_nns[i](proj_inputs[:, i])
            for i in range(self.num_networks)
        ]
        return individual_outputs, proj_inputs

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        individual_outputs, proj_inputs = self.calc_outputs(inputs)
        stacked_out = torch.stack(individual_outputs, dim=-1) * self.weight
        dropout_out = self.dropout_layer(stacked_out)
        out = torch.sum(dropout_out, dim=-1) + self.bias
        return out, dropout_out, proj_inputs, self.weight, self.bias
