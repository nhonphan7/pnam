from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def reg_penalty(
    model: nn.Module,
    fnn_out: torch.Tensor,
    weight: torch.Tensor,
    rot_reg_mode: str,
    rot_reg: float,
    proj_reg: float,
    weight_reg: float,
    output_reg: float,
    l2_reg: float,
    verbose: bool
) -> torch.Tensor:
    """Compute penalized loss.

    Args:
        model: Neural network model.
        fnn_out: Outputs computed by feature networks.
        weight: Weights of output layer.
        rot_reg_mode: 'svd' or 'ortho'.
        rot_reg: Coefficient for L2 regularization of rotation matrices.
        proj_reg: Coefficient for L1 regularization of projection matrix.
        weight_reg: Coefficient for L1 regularization of output weights.
        output_reg: Coefficient for L2 regularization of `FeatureNN` outputs.
        l2_reg: Coefficient for L2 regularization of `FeatureNN` parameters.
        verbose: Whether to print each individual regularization.
    """
    def rots_loss(proj_mat: torch.Tensor) -> torch.Tensor:
        """Penalize L2 norm of rotation matrices from SVD."""
        # Clamp and add epsilon to keep `sqrt` differentiable at minimum
        eps = 1e-12
        U, _, Vh = torch.linalg.svd(proj_mat)
        U_size, Vh_size = U.size(0), Vh.size(0)
        U_loss = torch.sqrt(
            torch.clamp(2. * (U_size - torch.trace(U)), min=0.) + eps
        )
        Vh_loss = torch.sqrt(
            torch.clamp(2. * (Vh_size - torch.trace(Vh)), min=0.) + eps
        )
        return (U_loss + Vh_loss) / U_size

    def ortho_loss(proj_mat: torch.Tensor) -> torch.Tensor:
        """Penalize L2 norm of off-diagonal of Gram matrix."""
        gram = proj_mat.T @ proj_mat
        off_diag = gram - torch.diag(torch.diagonal(gram))
        return torch.sqrt(torch.sum(off_diag**2)) / proj_mat.size(0)

    def proj_sparsity(proj_mat: torch.Tensor) -> torch.Tensor:
        """Penalize L1 norm of projection matrix."""
        return torch.sum(torch.abs(proj_mat)) / proj_mat.size(0)

    def group_lasso(weight: torch.Tensor) -> torch.Tensor:
        """Penalize L1 norm of output weights."""
        return torch.sum(torch.abs(weight)) / weight.size(1)

    def features_loss(fnn_out: torch.Tensor) -> torch.Tensor:
        """Penalize L2 norm of `FeatureNN` outputs."""
        return torch.sqrt(torch.sum(fnn_out**2)) / fnn_out.size(0)

    def weight_decay(model: nn.Module, fnn_out: torch.Tensor) -> torch.Tensor:
        """Penalize L2 norm of `FeatureNN` parameters."""
        l2_loss = 0.
        for x in model.feature_nns.parameters():
            l2_loss += torch.sum(x**2)
        return torch.sqrt(l2_loss) / fnn_out.size(0)

    # Get projection matrix
    proj_mat = None
    if hasattr(model, 'linear'):
        proj_mat = model.linear.weight

    reg_loss = 0.
    if proj_mat is not None:
        if rot_reg > 0:
            if rot_reg_mode == 'svd':
                rot_fn = rots_loss
            elif rot_reg_mode == 'ortho':
                rot_fn = ortho_loss
            else:
                raise NameError(
                    'This rotation regularization mode is not implemented.'
                )
            rot_constraint = rot_reg * rot_fn(proj_mat)
            reg_loss += rot_constraint
            if verbose:
                print(
                    '\nRotation regularization: '
                    f'{rot_constraint.detach().cpu().numpy().item():.10f}'
                )
        if proj_reg > 0:
            proj_constraint = proj_reg * proj_sparsity(proj_mat)
            reg_loss += proj_constraint
            if verbose:
                print(
                    'Projection regularization: '
                    f'{proj_constraint.detach().cpu().numpy().item():.10f}'
                )
    if weight is not None and weight_reg > 0:
        weight_constraint = weight_reg * group_lasso(weight)
        reg_loss += weight_constraint
        if verbose:
            print(
                '\nWeight regularization: '
                f'{weight_constraint.detach().cpu().numpy().item():.10f}'
            )
    if fnn_out is not None:
        if output_reg > 0:
            output_constraint = output_reg * features_loss(fnn_out)
            reg_loss += output_constraint
            if verbose:
                print(
                    '\nOutput regularization: '
                    f'{output_constraint.detach().cpu().numpy().item():.10f}'
                )
        if l2_reg > 0:
            l2_constraint = l2_reg * weight_decay(model, fnn_out)
            reg_loss += l2_constraint
            if verbose:
                print(
                    'L2 regularization: '
                    f'{l2_constraint.detach().cpu().numpy().item():.10f}'
                )
    return reg_loss


def make_penalized_loss_func(
    loss_func: Callable,
    regression: bool,
    num_outputs: int,
    num_targets: int,
    rot_reg_mode: str,
    rot_reg: float,
    proj_reg: float,
    weight_reg: float,
    output_reg: float,
    l2_reg: float,
    verbose: bool
) -> torch.Tensor:
    def penalized_loss_func(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module = None,
        fnn_out: torch.Tensor = None,
        weight: torch.Tensor = None
    ) -> torch.Tensor:
        loss = loss_func(predictions, targets)
        loss += reg_penalty(
            model,
            fnn_out,
            weight,
            rot_reg_mode,
            rot_reg,
            proj_reg,
            weight_reg,
            output_reg,
            l2_reg,
            verbose
        )
        return loss

    if not loss_func:
        if regression:
            loss_func = F.mse_loss
        else:
            if num_outputs == num_targets:
                loss_func = F.binary_cross_entropy_with_logits
            else:
                loss_func = F.cross_entropy
    return penalized_loss_func
