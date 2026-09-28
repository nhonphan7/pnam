import datetime
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torchvision import datasets

sys.path.append('../')
from pnam.utils.plot import *
from pnam.wrapper import PNAMBase


def plot_digits(dataset, digit_indices, prune_indices, plot_dir):
    X = dataset.data.view(-1, 28 * 28).to(dtype=torch.float).numpy()
    X[:, prune_indices] = np.nan
    X = X.reshape(-1, 28, 28)
    _, axes = plt.subplots(2, 5, figsize=(14, 6))
    axes = axes.flatten()
    for i, digit_idx in enumerate(digit_indices):
        axes[i].imshow(X[digit_idx])
        axes[i].set(xticks=[], yticks=[])
    plt.tight_layout()
    if plot_dir:
        plt.savefig(f'{plot_dir}/digits{784 - prune_indices.shape[0]}.png')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    time = datetime.datetime.now().isoformat()
    # time = 'mnist_pnam64'

    small_size = 20
    large_size = 24
    plt.rc('font', size=small_size)
    plt.rc('axes', labelsize=large_size)

    train_dataset = datasets.MNIST('../../data', train=True, download=True)
    test_dataset = datasets.MNIST('../../data', train=False)

    X_train = train_dataset.data.view(-1, 28 * 28).to(dtype=torch.float)
    X_test = test_dataset.data.view(-1, 28 * 28).to(dtype=torch.float)
    y_train = train_dataset.targets
    y_test = test_dataset.targets

    random_state = 42
    pnam = True
    num_inputs = X_train.shape[1]
    proj_size = 64
    num_networks = proj_size if proj_size > 0 else num_inputs
    num_outputs = 10
    metric = 'acc'
    sobolev = False

    model_idx = 0
    weight_thresh = 0.001
    proj_mat_idx = 400

    model_pnam = PNAMBase(
        random_state=random_state,
        num_learners=1,
        pnam=pnam,
        std=0.,
        proj_size=proj_size,
        hidden_sizes=[64, 32],
        num_outputs=num_outputs,
        activation=F.silu,
        dropout=0.,
        feature_dropout=0.,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        regression=False,
        rot_reg=0.01,
        proj_reg=0.01,
        weight_reg=0.01,
        output_reg=0.01,
        l2_reg=0.01,
        verbose=False,
        metric=metric,
        scale=False,
        sobolev=sobolev,
        val_split=0.2,
        n_jobs=1,
        batch_size=256,
        log_dir=f'{time}/pnam',
        lr=0.001,
        decay_step=1,
        decay_rate=0.995,
        num_epochs=1000,
        energy=False,
        patience=50
    )

    model_pnam.fit(X_train, y_train)
    # model_pnam.load_checkpoints(f'{time}/pnam')

    plot_dir = f'{time}/figures/{model_idx}'
    os.makedirs(plot_dir, exist_ok=True)
    print(f'Model index: {model_idx}')

    plot_loss_or_metric(
        time, model_idx, 'loss', semilogy=False, plot_dir=plot_dir
    )
    if metric:
        plot_loss_or_metric(
            time, model_idx, metric, semilogy=False, plot_dir=plot_dir
        )

    (
        preds_test,
        feats_out_test,
        feats_in_test,
        weight,
        bias,
        proj_mat,
        grad_in_test
    ) = model_pnam.predict(X_test, y_test, model_idx)

    if pnam:
        weight_mean = np.mean(abs(weight), axis=0)

        if proj_size > 0:
            for i in range(proj_size):
                if weight_mean[i] == 0:
                    proj_mat[i] = 0.

            proj_mat_mean = np.mean(abs(proj_mat), axis=0)

            top_n_acc = []
            proj_mat_zero = np.copy(proj_mat)
            sort_indices = np.argsort(proj_mat_mean)
            for i in range(num_inputs):
                print(f'Top n inputs: {num_inputs - i}')
                proj_mat_zero[:, sort_indices[:i]] = 0.
                (
                    preds_test,
                    feats_out_test,
                    feats_in_test,
                    _,
                    _,
                    _,
                    grad_in_test
                ) = model_pnam.predict(
                    X_test, y_test, model_idx, weight, proj_mat_zero
                )
                preds_test = np.argmax(preds_test, axis=-1)
                num = (preds_test == y_test.detach().cpu().numpy()).sum()
                denom = len(preds_test)
                acc = num / denom
                top_n_acc.append(acc)
            top_n_acc = np.array(top_n_acc)
            plot_top_n_acc(top_n_acc, graph='line', plot_dir=plot_dir)

    if pnam:
        weight_zero = np.copy(weight)
        weight_zero[abs(weight_zero) < weight_thresh] = 0.
        weight_zero_mean = np.mean(abs(weight_zero), axis=0)

        if proj_size > 0:
            proj_mat_zero = np.copy(proj_mat)
            for i in range(proj_size):
                if weight_zero_mean[i] == 0:
                    proj_mat_zero[i] = 0.

            proj_mat_zero[:, sort_indices[:-proj_mat_idx]] = 0.

        (
            preds_test, feats_out_test, feats_in_test, _, _, _, grad_in_test
        ) = model_pnam.predict(
            X_test,
            y_test,
            model_idx,
            weight_zero,
            proj_mat_zero if proj_size > 0 else None
        )

        plot_digits(
            test_dataset,
            digit_indices=[3, 2, 1, 18, 4, 8, 11, 0, 61, 7],
            prune_indices=sort_indices[:-proj_mat_idx],
            plot_dir=plot_dir
        )
