import torch
import numpy as np
import sklearn.metrics as sk_metrics
from abc import abstractmethod


class Metric:

    def __init__(self) -> None:
        pass

    @abstractmethod
    def update(self) -> None:
        pass

    @abstractmethod
    def compute(self) -> float:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class Accuracy(Metric):

    def __init__(self) -> None:
        self.num = 0
        self.denom = 0
        self.updated = False

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        # TODO: Exception handling/input checking
        num_outputs = 1 if len(predictions.size()) == 1 else predictions.size(1)
        num_targets = 1 if len(targets.size()) == 1 else targets.size(1)
        if num_outputs == num_targets:
            predictions = torch.round(torch.sigmoid(predictions))
        else:
            predictions = torch.argmax(predictions, dim=-1)
        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        self.num += (predictions == targets).sum()
        # Count elements rather than rows to average over multilabel outputs
        self.denom += predictions.size
        self.updated = True

    def compute(self) -> float:
        if not self.updated:
            # TODO: Find appropriate exception
            raise Exception()
        return self.num / self.denom

    def reset(self) -> None:
        self.num = 0
        self.denom = 0
        self.updated = False


class AUC(Metric):

    def __init__(self) -> None:
        self.predictions = []
        self.targets = []
        self.updated = False

    @abstractmethod
    def score_func(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        pass

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        # TODO: Exception handling/input checking
        self.predictions.append(predictions)
        self.targets.append(targets)
        self.updated = True

    def compute(self) -> float:
        if not self.updated:
            # TODO: Find appropriate exception
            raise Exception()
        
        predictions = torch.cat(self.predictions).detach().cpu().numpy()
        targets = torch.cat(self.targets).detach().cpu().numpy()
        return self.score_func(predictions, targets)

    def reset(self) -> None:
        self.predictions = []
        self.targets = []
        self.updated = False


class AUROC(AUC):

    def __init__(self) -> None:
        super(AUROC, self).__init__()

    def score_func(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        return sk_metrics.roc_auc_score(targets, predictions)


class AveragePrecision(AUC):

    def __init__(self) -> None:
        super(AveragePrecision, self).__init__()

    def score_func(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        return sk_metrics.average_precision_score(targets, predictions)


class MeanError(Metric):

    def __init__(self) -> None:
        self.sum_of_errors = 0.
        self.num_examples = 0
        self.updated = False

    @abstractmethod
    def distance_func(predictions: np.ndarray, targets: np.ndarray) -> float:
        pass

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        # TODO: Exception handling/input checking
        predictions = predictions.detach().cpu().numpy() 
        targets = targets.detach().cpu().numpy()
        self.sum_of_errors += self.distance_func(predictions, targets)
        # Count elements rather than rows to average over multiple outputs
        self.num_examples += predictions.size
        self.updated = True

    def compute(self) -> float:
        if not self.updated:
            # TODO: Find appropriate exception
            raise Exception()
        return self.sum_of_errors / self.num_examples

    def reset(self) -> None:
        self.sum_of_errors = 0.
        self.num_examples = 0
        self.updated = False


class MeanSquaredError(MeanError):

    def __init__(self) -> None:
        super(MeanSquaredError, self).__init__()

    def distance_func(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> float:
        return ((predictions - targets)**2).sum()


class MeanAbsoluteError(MeanError):
    
    def __init__(self) -> None:
        super(MeanAbsoluteError, self).__init__()

    def distance_func(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> float:
        return (abs(predictions - targets)).sum()
