from typing import Tuple

import torch
import torch.nn as nn
from torch.autograd import grad


def forward_pass(
    features: torch.Tensor,
    model: nn.Module,
    pnam: bool,
    scale: bool,
    sobolev: bool,
    energy: bool,
    X_scale_: torch.Tensor,
    y_scale_: torch.Tensor,
    y_min_: torch.Tensor
) -> Tuple[torch.Tensor, ...]:
    fnn_out, fnn_in, weight, bias = None, None, None, None
    if pnam:
        predictions, fnn_out, fnn_in, weight, bias = model(features)
    else:
        predictions = model(features)

    predictions_grad = None
    if sobolev:
        jac = torch.stack([grad(
            torch.sum(predictions[:, i]), features, create_graph=True
        )[0] for i in range(predictions.size(1))], dim=1)
        preds_grad = jac[:, :, 0]

        if scale:
            if energy:
                grad_unscale = preds_grad * X_scale_[0] / y_scale_[:-1]
            else:
                grad_unscale = preds_grad * X_scale_[0]
            grad_sum = torch.sum(grad_unscale, dim=1, keepdim=True)
            predictions_grad = grad_sum * y_scale_[-1] + y_min_[-1]
        else:
            predictions_grad = torch.sum(preds_grad, dim=1, keepdim=True)
    return predictions, fnn_out, fnn_in, weight, bias, predictions_grad


def squeeze_last(tensor: torch.Tensor) -> torch.Tensor:
    """Squeeze trailing singleton dimension, preserving batch dimension."""
    return tensor.squeeze(-1) if len(tensor.size()) > 1 else tensor
