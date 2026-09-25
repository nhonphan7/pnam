from typing import Tuple, Union

import torch
import numpy as np
import pandas as pd


class PNAMDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        y: Union[np.ndarray, pd.DataFrame, torch.Tensor]
    ) -> None:
        """Dataset for projected neural additive models.

        Args:
            X: Feature array.
            y: Target array.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()

        if isinstance(X, np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float, requires_grad=True)
        elif isinstance(X, torch.Tensor):
            self.X = X
        if isinstance(y, np.ndarray):
            self.y = torch.tensor(y, dtype=torch.float)
        elif isinstance(y, torch.Tensor):
            self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, indices: list) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[indices], self.y[indices]
