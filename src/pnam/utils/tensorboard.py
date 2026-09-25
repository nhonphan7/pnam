"""Utility class for logging to TensorBoard."""

from typing import Any, Mapping

import os
from torch.utils.tensorboard import SummaryWriter


def format_key(key: str) -> str:
    """Internal function for formatting keys in TensorBoard format."""
    return key.title().replace('_', '')


class TensorBoardLogger:
    """A simple PyTorch-friendly TensorBoard wrapper."""

    def __init__(self, label: str = 'Logs', log_dir: str = 'output') -> None:
        """Construct a simple TensorBoard wrapper.

        Args:
          label: Label string to use when logging. Default to 'Logs'.
          log_dir: Directory where log files are stored.
        """
        self.label = label

        # Make sure output directory exists
        self.log_dir = os.path.join(log_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize TensorBoard writer
        self.summary_writer = SummaryWriter(log_dir=self.log_dir)

    def write(self, data: Mapping[str, Any], epoch: int) -> None:
        for key, value in data.items():
            self.summary_writer.add_scalar(
                f'{self.label}/{format_key(key)}', value, global_step=epoch
            )
