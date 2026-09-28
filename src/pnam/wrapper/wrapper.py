from typing import Callable, Sequence, Tuple, Union

import os
import random
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from joblib import load
from sklearn.exceptions import NotFittedError
from torch.autograd import grad

from pnam.data import PNAMDataset
from pnam.models import Checkpointer, FC, PNAM
from pnam.trainer import Trainer
from pnam.trainer.forward import forward_pass, squeeze_last
from pnam.trainer.losses import make_penalized_loss_func


class PNAMBase:

    def __init__(
        self,
        random_state: int = 42,
        num_learners: int = 1,
        pnam: bool = True,
        std: float = 0.,
        proj_size: int = 8,
        hidden_sizes: Sequence[int] = (64, 32),
        num_outputs: int = 1,
        activation: Callable = F.silu,
        dropout: float = 0.,
        feature_dropout: float = 0.,
        device: str = 'cpu',
        loss_func: Callable = None,
        regression: bool = True,
        rot_reg_mode: str = 'svd',
        rot_reg: float = 0.001,
        proj_reg: float = 0.001,
        weight_reg: float = 0.001,
        output_reg: float = 0.001,
        l2_reg: float = 0.001,
        verbose: bool = False,
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
        self.random_state = random_state
        self.num_learners = num_learners
        self.pnam = pnam
        self.std = std
        self.proj_size = proj_size
        self.hidden_sizes = hidden_sizes
        self.num_outputs = num_outputs
        self.activation = activation
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        self.device = device
        self.loss_func = loss_func
        self.regression = regression
        self.rot_reg_mode = rot_reg_mode
        self.rot_reg = rot_reg
        self.proj_reg = proj_reg
        self.weight_reg = weight_reg
        self.output_reg = output_reg
        self.l2_reg = l2_reg
        self.verbose = verbose
        self.metric = metric
        self.scale = scale
        self.sobolev = sobolev
        self.val_split = val_split
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.log_dir = log_dir
        self.lr = lr
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.num_epochs = num_epochs
        self.energy = energy
        self.save_model_frequency = save_model_frequency
        self.monitor_loss = monitor_loss
        self.early_stop_mode = early_stop_mode
        self.patience = patience

        self.best_checkpoint_suffix = 'best'
        self.fitted = False

    def set_random_state(self) -> None:
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

    def initialize_models(self) -> None:
        self.models = []
        for _ in range(self.num_learners):
            if self.pnam:
                model = PNAM(
                    std=self.std,
                    num_inputs=self.num_inputs,
                    proj_size=self.proj_size,
                    hidden_sizes=self.hidden_sizes,
                    num_outputs=self.num_outputs,
                    activation=self.activation,
                    dropout=self.dropout,
                    feature_dropout=self.feature_dropout
                )
            else:
                model = FC(
                    std=self.std,
                    num_inputs=self.num_inputs,
                    hidden_sizes=self.hidden_sizes,
                    num_outputs=self.num_outputs,
                    activation=self.activation,
                    dropout=self.dropout
                )
            self.models.append(model)

    def models_to_device(self, device: str) -> None:
        for model in self.models:
            model.to(device)

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        y: Union[np.ndarray, pd.DataFrame, torch.Tensor]
    ) -> None:
        self.num_inputs = X.shape[1]

        self.set_random_state()
        if not self.fitted:
            self.initialize_models()

        self.fitted = False
        self.models_to_device(self.device)

        self.partial_fit(X, y)

    def partial_fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        y: Union[np.ndarray, pd.DataFrame, torch.Tensor]
    ) -> None:
        dataset = PNAMDataset(X, y)
        self.num_targets = (
            1 if len(dataset.y.size()) == 1 else dataset.y.size(1)
        )

        if self.fitted:
            self.X = dataset.X
            self.y = dataset.y

        self.criterion = make_penalized_loss_func(
            self.loss_func,
            self.regression,
            self.num_outputs,
            self.num_targets,
            self.rot_reg_mode,
            self.rot_reg,
            self.proj_reg,
            self.weight_reg,
            self.output_reg,
            self.l2_reg,
            self.verbose
        )

        self.trainer = Trainer(
            dataset=dataset,
            models=self.models,
            criterion=self.criterion,
            random_state=self.random_state,
            num_learners=self.num_learners,
            pnam=self.pnam,
            device=self.device,
            metric=self.metric,
            scale=self.scale,
            sobolev=self.sobolev,
            val_split=self.val_split,
            n_jobs=self.n_jobs,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            log_dir=self.log_dir,
            lr=self.lr,
            decay_step=self.decay_step,
            decay_rate=self.decay_rate,
            num_epochs=self.num_epochs,
            energy=self.energy,
            save_model_frequency=self.save_model_frequency,
            monitor_loss=self.monitor_loss,
            early_stop_mode=self.early_stop_mode,
            patience=self.patience
        )

        if not self.fitted:
            self.trainer.train_ensemble()
            self.trainer.close()

            # Move models to 'cpu' so predictions can be made on 'cpu' data
            self.models_to_device('cpu')

            self.fitted = True

    def load_checkpoints(self, checkpoint_dir: str) -> None:
        self.models = []
        for i in range(self.num_learners):
            checkpointer = Checkpointer(log_dir=os.path.join(
                checkpoint_dir, str(i)
            ))
            model = checkpointer.load(self.best_checkpoint_suffix)
            model.eval()
            self.models.append(model)

        self.fitted = True

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        y: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        model_idx: int = 0,
        weight: np.ndarray = None,
        proj_mat: np.ndarray = None
    ) -> Tuple[np.ndarray, ...]:
        if self.fitted:
            self.partial_fit(X, y)
        else:
            raise NotFittedError(
                'This model instance is not fitted yet. Call `fit` with '
                'appropriate arguments before using this method.'
            )

        X_scale_, y_scale_, y_min_ = None, None, None
        if self.scale and self.sobolev:
            X_scaler = load(f'{self.log_dir}/scaler/X_scaler.pkl')
            X_scale_ = torch.tensor(X_scaler.scale_, dtype=torch.float)
            y_scaler = load(f'{self.log_dir}/scaler/y_scaler.pkl')
            y_scale_ = torch.tensor(y_scaler.scale_, dtype=torch.float)
            y_min_ = torch.tensor(y_scaler.min_, dtype=torch.float)

        model = self.models[model_idx]
        model.eval()
        # `weight` and `proj_mat` permanently overwrite model's parameters
        if weight is not None:
            model.weight = torch.nn.Parameter(torch.tensor(
                weight, dtype=torch.float
            ))
        if proj_mat is not None:
            model.linear.weight = torch.nn.Parameter(torch.tensor(
                proj_mat, dtype=torch.float
            ))

        preds, feats_out, feats_in, weight, bias, preds_grad = forward_pass(
            self.X,
            model,
            self.pnam,
            self.scale,
            self.sobolev,
            self.energy,
            X_scale_,
            y_scale_,
            y_min_
        )

        proj_mat, grad_in = None, None
        if self.pnam:
            if self.proj_size > 0:
                proj_mat = model.linear.weight
            if self.sobolev:
                grad_in = torch.stack([grad(
                    torch.sum(preds[:, i]), feats_in, create_graph=True
                )[0] for i in range(preds.size(1))], dim=1)

        predictions = preds
        if self.sobolev and not self.energy:
            predictions = preds_grad
        elif self.sobolev and self.energy:
            predictions = torch.cat((preds, preds_grad), dim=-1)

        predictions, y = squeeze_last(predictions), squeeze_last(self.y)
        loss = self.criterion(predictions, y)
        print(f'Loss: {loss.detach().cpu().numpy().item():.10f}')
        if self.metric:
            metric = self.trainer.create_metric()
            self.trainer.update_metric(metric, predictions, y)
            print(
                f'{self.trainer.metric_name.title()}: '
                f'{metric.compute():.10f}'
            )

        predictions = predictions.detach().cpu().numpy()
        if self.pnam:
            feats_out = feats_out.detach().cpu().numpy()
            feats_in = feats_in.detach().cpu().numpy()
            weight = weight.detach().cpu().numpy()
            bias = bias.detach().cpu().numpy()
            if self.proj_size > 0:
                proj_mat = proj_mat.detach().cpu().numpy()
            if self.sobolev:
                grad_in = grad_in.detach().cpu().numpy()
        return (
            predictions, feats_out, feats_in, weight, bias, proj_mat, grad_in
        )
