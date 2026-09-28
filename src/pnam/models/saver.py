"""Utility class for saving model checkpoints."""

import inspect
import os
import torch
import torch.nn as nn


class Checkpointer:
    """A simple PyTorch save/load wrapper."""

    def __init__(self, device: str = 'cpu', log_dir: str = 'output') -> None:
        """Construct a simple save/load `Checkpointer`."""
        self.device = device
        self.ckpt_dir = os.path.join(log_dir, 'ckpts')
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def save(self, model: nn.Module, epoch: int) -> str:
        """Save model to `ckpt_dir/model-epoch.pt` file."""
        ckpt_path = os.path.join(self.ckpt_dir, f'model-{epoch}.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'attributes': vars(model),
            'class': type(model)
        }, ckpt_path)
        return ckpt_path

    def load(self, epoch: int) -> nn.Module:
        """Load model from `ckpt_dir/model-epoch.pt` file."""
        ckpt_path = os.path.join(self.ckpt_dir, f'model-{epoch}.pt')
        ckpt = torch.load(
            ckpt_path, map_location=self.device, weights_only=False
        )
        constructor = ckpt['class']
        constructor_args = inspect.getfullargspec(constructor).args
        args = {
            k: v for k, v in ckpt['attributes'].items()
            if k in constructor_args
        }
        model = constructor(**args)
        model.load_state_dict(ckpt['model_state_dict'])
        return model
