from typing import Callable, Sequence, Tuple

import gc
import os
import torch
import torch.nn as nn
import torch.optim as optim
from joblib import delayed, load, Parallel
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import DataLoader, Subset
from tqdm.autonotebook import tqdm

from pnam.models import Checkpointer
from pnam.trainer.forward import forward_pass, squeeze_last
from pnam.trainer.metrics import *
from pnam.utils import TensorBoardLogger


class Trainer:

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        models: Sequence[nn.Module],
        criterion: Callable,
        random_state: int = 42,
        num_learners: int = 1,
        pnam: bool = True,
        device: str = 'cpu',
        metric: str = None,
        scale: bool = True,
        sobolev: bool = False,
        val_split: float = 0.2,
        n_jobs: int = None,
        batch_size: int = 256,
        num_workers: int = 0,
        log_dir: str = None,
        lr: float = 0.001,
        decay_step: int = 1,
        decay_rate: float = 0.995,
        num_epochs: int = 1000,
        energy: bool = False,
        save_model_frequency: int = 10,
        monitor_loss: bool = True,
        early_stop_mode: str = 'min',
        patience: int = 50
    ) -> None:
        self.dataset = dataset
        self.models = models
        self.criterion = criterion
        self.random_state = random_state
        self.num_learners = num_learners
        self.pnam = pnam
        self.device = device
        self.metric_name = metric.lower() if metric else None
        self.scale = scale
        self.sobolev = sobolev
        self.val_split = val_split
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.num_epochs = num_epochs
        self.energy = energy
        self.save_model_frequency = save_model_frequency
        self.monitor_loss = monitor_loss
        self.early_stop_mode = early_stop_mode
        self.patience = patience
        # Disable `tqdm` if concurrency > 1
        self.disable_tqdm = self.n_jobs not in (None, 1)

        self.log_dir = log_dir
        if not self.log_dir:
            self.log_dir = 'output'

        self.best_checkpoint_suffix = 'best'

        self.X_scale_, self.X_min_ = None, None
        self.y_scale_, self.y_min_ = None, None
        if self.scale and self.sobolev:
            X_scaler = load(f'{self.log_dir}/scaler/X_scaler.pkl')
            self.X_scale_ = torch.tensor(
                X_scaler.scale_, dtype=torch.float, device=self.device
            )
            self.X_min_ = torch.tensor(
                X_scaler.min_, dtype=torch.float, device=self.device
            )
            y_scaler = load(f'{self.log_dir}/scaler/y_scaler.pkl')
            self.y_scale_ = torch.tensor(
                y_scaler.scale_, dtype=torch.float, device=self.device
            )
            self.y_min_ = torch.tensor(
                y_scaler.min_, dtype=torch.float, device=self.device
            )

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        model: nn.Module,
        optimizer: optim.Optimizer,
        metric: Metric
    ) -> torch.Tensor:
        """Perform a single gradient descent optimization step."""
        features, targets = [t.to(self.device) for t in batch]
        if self.sobolev:
            X_ref = torch.zeros((1, features.size(1)), device=self.device)
            X_ref = X_ref * self.X_scale_ + self.X_min_
            features = torch.cat((features, X_ref))
            y_ref = torch.zeros((1, targets.size(1)), device=self.device)
            y_ref = y_ref * self.y_scale_ + self.y_min_
            targets = torch.cat((targets, y_ref))

        # Reset optimizer's gradients
        optimizer.zero_grad()

        # Forward pass of model
        preds, fnn_out, _, weight, _, preds_grad = forward_pass(
            features,
            model,
            self.pnam,
            self.scale,
            self.sobolev,
            self.energy,
            self.X_scale_,
            self.y_scale_,
            self.y_min_
        )

        predictions = preds
        if self.sobolev and not self.energy:
            predictions = preds_grad
        elif self.sobolev and self.energy:
            predictions = torch.cat((preds, preds_grad), dim=-1)

        predictions, targets = squeeze_last(predictions), squeeze_last(targets)
        loss = self.criterion(predictions, targets, model, fnn_out, weight)
        self.update_metric(metric, predictions, targets)

        # Backward pass
        loss.backward()

        # Perform a gradient descent step
        optimizer.step()
        return loss

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        model: nn.Module,
        optimizer: optim.Optimizer,
        metric: Metric
    ) -> Tuple[torch.Tensor, float]:
        """Perform an epoch of gradient descent on `DataLoader`."""
        model.train()
        loss = 0.
        with tqdm(dataloader, leave=False, disable=self.disable_tqdm) as pbar:
            for batch in pbar:
                # Perform a gradient descent step
                step_loss = self.train_step(batch, model, optimizer, metric)
                loss += step_loss.detach()

        metric_train = None
        if metric:
            metric_train = metric.compute()
            metric.reset()
        return loss / len(dataloader), metric_train

    def evaluate_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        model: nn.Module,
        metric: Metric
    ) -> torch.Tensor:
        """Evaluate model on a mini-batch."""
        features, targets = [t.to(self.device) for t in batch]
        if self.sobolev:
            X_ref = torch.zeros((1, features.size(1)), device=self.device)
            X_ref = X_ref * self.X_scale_ + self.X_min_
            features = torch.cat((features, X_ref))
            y_ref = torch.zeros((1, targets.size(1)), device=self.device)
            y_ref = y_ref * self.y_scale_ + self.y_min_
            targets = torch.cat((targets, y_ref))

        # Forward pass of model
        preds, fnn_out, _, weight, _, preds_grad = forward_pass(
            features,
            model,
            self.pnam,
            self.scale,
            self.sobolev,
            self.energy,
            self.X_scale_,
            self.y_scale_,
            self.y_min_
        )

        predictions = preds
        if self.sobolev and not self.energy:
            predictions = preds_grad
        elif self.sobolev and self.energy:
            predictions = torch.cat((preds, preds_grad), dim=-1)

        # Calculate loss on a mini-batch
        predictions, targets = squeeze_last(predictions), squeeze_last(targets)
        loss = self.criterion(predictions, targets, model, fnn_out, weight)
        self.update_metric(metric, predictions, targets)
        return loss

    def evaluate_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        model: nn.Module,
        metric: Metric
    ) -> Tuple[torch.Tensor, float]:
        """Perform an evaluation of model on `DataLoader`."""
        model.eval()
        loss = 0.
        with tqdm(dataloader, leave=False, disable=self.disable_tqdm) as pbar:
            for batch in pbar:
                # Accumulate loss in dataset
                # Only Sobolev constraint requires gradients in evaluation
                if self.sobolev:
                    step_loss = self.evaluate_step(batch, model, metric)
                else:
                    with torch.no_grad():
                        step_loss = self.evaluate_step(batch, model, metric)
                loss += step_loss.detach()

        metric_val = None
        if metric:
            metric_val = metric.compute()
            metric.reset()
        return loss / len(dataloader), metric_val

    def train_ensemble(self) -> None:
        ss = ShuffleSplit(
            n_splits=self.num_learners,
            test_size=self.val_split,
            random_state=self.random_state
        )

        self.models[:] = Parallel(n_jobs=self.n_jobs)(
            delayed(self.train_learner)(i, train_indices, val_indices)
            for i, (train_indices, val_indices) in enumerate(
                ss.split(self.dataset.X, self.dataset.y)
            )
        )

    def train_learner(
        self,
        model_idx: int,
        train_indices: np.ndarray,
        val_indices: np.ndarray
    ) -> nn.Module:
        # Set random seed for each process to guarantee reproducibility
        torch.manual_seed(self.random_state + model_idx)

        model = self.models[model_idx]
        train_subset = Subset(self.dataset, train_indices)
        val_subset = Subset(self.dataset, val_indices)

        train_dl = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

        val_dl = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        log_subdir = os.path.join(self.log_dir, str(model_idx))
        writer = TensorBoardLogger(log_dir=log_subdir)
        checkpointer = Checkpointer(log_dir=log_subdir)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.decay_step, gamma=self.decay_rate
        )

        metric = self.create_metric()

        model = self.train(
            train_dl,
            val_dl,
            model,
            writer,
            checkpointer,
            optimizer,
            scheduler,
            metric
        )
        return model

    def train(
        self,
        train_dl: torch.utils.data.DataLoader,
        val_dl: torch.utils.data.DataLoader,
        model: nn.Module,
        writer: TensorBoardLogger,
        checkpointer: Checkpointer,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        metric: Metric
    ) -> nn.Module:
        """Train model for a specified number of epochs."""
        best_loss_or_metric = float('inf')
        epochs_since_best = 0

        with tqdm(
            range(self.num_epochs), disable=self.disable_tqdm
        ) as pbar_epoch:
            for epoch in pbar_epoch:
                # Train model on entire training dataset
                # Write to TensorBoard
                loss_train, metric_train = self.train_epoch(
                    train_dl, model, optimizer, metric
                )
                loss_train_epoch = loss_train.detach().cpu().numpy().item()
                writer.write({'loss_train_epoch': loss_train_epoch}, epoch)
                if metric:
                    writer.write({
                        f'{self.metric_name}_train_epoch': metric_train
                    }, epoch)

                # Evaluate model on entire validation dataset
                # Write to TensorBoard
                loss_val, metric_val = self.evaluate_epoch(
                    val_dl, model, metric
                )
                loss_val_epoch = loss_val.detach().cpu().numpy().item()
                writer.write({'loss_val_epoch': loss_val_epoch}, epoch)
                if metric:
                    writer.write({
                        f'{self.metric_name}_val_epoch': metric_val
                    }, epoch)

                scheduler.step()

                # Print loss and metric
                print(
                    f'\nEpoch({epoch}):'
                    '\n\tTraining loss: '
                    f'{loss_train.detach().cpu().numpy().item():.10f}'
                    '\n\tValidation loss: '
                    f'{loss_val.detach().cpu().numpy().item():.10f}'
                )
                if metric:
                    print(
                        f'\n\tTraining {self.metric_name}: {metric_train:.10f}'
                        f'\n\tValidation {self.metric_name}: {metric_val:.10f}'
                    )

                # Save model checkpoint
                if (
                    self.save_model_frequency > 0
                    and epoch % self.save_model_frequency == 0
                ):
                    checkpointer.save(model, epoch)

                # Save best checkpoint for early stopping
                loss_or_metric = loss_val if self.monitor_loss else metric_val
                if self.early_stop_mode == 'max':
                    loss_or_metric = -loss_or_metric

                if self.patience > 0 and loss_or_metric < best_loss_or_metric:
                    best_loss_or_metric = loss_or_metric
                    epochs_since_best = 0
                    checkpointer.save(model, self.best_checkpoint_suffix)

                # Stop training if early stopping patience is exceeded
                epochs_since_best += 1
                if self.patience > 0 and epochs_since_best > self.patience:
                    return checkpointer.load(self.best_checkpoint_suffix)

            # Return best checkpoint if early stopping is enabled
            if self.patience > 0 and best_loss_or_metric < float('inf'):
                return checkpointer.load(self.best_checkpoint_suffix)

            # If early stopping is not used, save last checkpoint as best
            checkpointer.save(model, self.best_checkpoint_suffix)
            return model

    def close(self) -> None:
        del self.dataset
        gc.collect()

    def create_metric(self) -> Metric:
        if not self.metric_name:
            return None
        elif self.metric_name == 'acc':
            return Accuracy()
        elif self.metric_name == 'auroc':
            return AUROC()
        elif self.metric_name == 'ap':
            return AveragePrecision()
        elif self.metric_name == 'mse':
            return MeanSquaredError()
        elif self.metric_name == 'mae':
            return MeanAbsoluteError()
        else:
            raise NameError('This metric is not implemented.')

    def update_metric(
        self, metric: Metric, predictions: torch.Tensor, targets: torch.Tensor
    ) -> None:
        if metric:
            metric.update(predictions, targets)
