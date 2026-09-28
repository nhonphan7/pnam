import datetime
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F

sys.path.append('../')
from pnam.utils.plot import *
from pnam.wrapper import PNAMBase


if __name__ == '__main__':
    time = datetime.datetime.now().isoformat()
    # time = 'knot_theory_classif_pnam8'

    small_size = 16
    large_size = 24
    plt.rc('font', size=small_size)
    plt.rc('axes', labelsize=large_size)

    # Download data: https://colab.research.google.com/github/deepmind/mathematics_conjectures/blob/main/knot_theory.ipynb#scrollTo=l10N2ZbHu6Ob
    data_path = '../../data/knot_theory_invariants.csv'
    df = pd.read_csv(data_path)
    X = df[df.keys()[1:-1]].to_numpy()
    y = df['signature'].to_numpy()

    X_labels = [
        'Adjoint torsion degree',
        'Torsion degree',
        'Re(short geodesic)',
        'Im(short geodesic)',
        'Injectivity radius',
        'Chern-Simons',
        'Cusp volume',
        'Longitudinal translation',
        'Im(meridional translation)',
        'Re(meridional translation)',
        'Volume',
        'Symmetry: $0$',
        'Symmetry: $D_3$',
        'Symmetry: $D_4$',
        'Symmetry: $D_6$',
        'Symmetry: $D_8$',
        r'Symmetry: $\frac{Z}{2} + \frac{Z}{2}$'
    ]
    # plot_vars(X, y, X_labels=X_labels, y_labels=['Signature'])

    # Normalize X
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std

    # Normalize y
    min_signature = y.min()
    max_signature = y.max()
    y = ((y - min_signature) / 2).astype(int)
    num_classes = int((max_signature - min_signature) / 2 + 1)

    random_state = 42
    train_ratio = 0.8
    num_points = X.shape[0]
    np.random.seed(random_state)
    train_indices = np.random.choice(
        num_points, int(train_ratio * num_points), replace=False
    )
    test_indices = np.array(list(set(range(num_points)) - set(train_indices)))

    X_train = torch.tensor(X[train_indices], dtype=torch.float)
    X_test = torch.tensor(X[test_indices], dtype=torch.float)
    y_train = torch.tensor(y[train_indices], dtype=torch.long)
    y_test = torch.tensor(y[test_indices], dtype=torch.long)

    pnam = True
    num_inputs = X.shape[1]
    proj_size = 8
    num_networks = proj_size if proj_size > 0 else num_inputs
    num_outputs = num_classes
    metric = 'acc'
    sobolev = False

    model_idx = 4
    weight_thresh = 0.01
    proj_mat_idx = 2

    model_pnam = PNAMBase(
        random_state=random_state,
        num_learners=10,
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
        scale=True,
        sobolev=sobolev,
        val_split=0.2,
        n_jobs=4,
        batch_size=256,
        log_dir=f'{time}/pnam',
        lr=0.001,
        decay_step=1,
        decay_rate=0.995,
        num_epochs=50,
        energy=False,
        patience=0  # Disable early stopping to keep last-epoch model
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
        plot_weight(
            weight,
            color=None,
            labels=[r'$\zeta_{,%s}$' % (i + 1) for i in range(num_networks)],
            colorbar=True,
            plot_dir=plot_dir,
            title='weight'
        )

        weight_mean = np.mean(abs(weight), axis=0)
        plot_mean(
            weight_mean,
            labels=['$f_{,%s}$' % (i + 1) for i in range(num_networks)],
            plot_dir=plot_dir,
            title='weight_mean'
        )

        if proj_size > 0:
            for i in range(proj_size):
                if weight_mean[i] == 0:
                    proj_mat[i] = 0.

            plot_weight(
                proj_mat,
                color=None,
                labels=['$T_{,%s}$' % (i + 1) for i in range(num_inputs)],
                colorbar=True,
                plot_dir=plot_dir,
                title='proj_mat'
            )

            proj_mat_mean = np.mean(abs(proj_mat), axis=0)
            plot_mean(
                proj_mat_mean,
                labels=X_labels,
                plot_dir=plot_dir,
                title='proj_mat_mean'
            )

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
            plot_top_n_acc(top_n_acc, graph='bar', plot_dir=plot_dir)

    if pnam:
        weight_zero = np.copy(weight)
        weight_zero[abs(weight_zero) < weight_thresh] = 0.
        weight_zero_mean = np.mean(abs(weight_zero), axis=0)

        plot_weight(
            weight_zero,
            color=None,
            labels=[r'$\zeta_{,%s}$' % (i + 1) for i in range(num_networks)],
            colorbar=True,
            plot_dir=plot_dir,
            title='weight_zero'
        )

        if proj_size > 0:
            proj_mat_zero = np.copy(proj_mat)
            for i in range(proj_size):
                if weight_zero_mean[i] == 0:
                    proj_mat_zero[i] = 0.

            proj_mat_zero[:, sort_indices[:-proj_mat_idx]] = 0.
            plot_weight(
                proj_mat_zero,
                color=None,
                labels=['$T_{,%s}$' % (i + 1) for i in range(num_inputs)],
                colorbar=True,
                plot_dir=plot_dir,
                title='proj_mat_zero'
            )

        (
            preds_test, feats_out_test, feats_in_test, _, _, _, grad_in_test
        ) = model_pnam.predict(
            X_test,
            y_test,
            model_idx,
            weight_zero,
            proj_mat_zero if proj_size > 0 else None
        )

        for i in range(num_outputs):
            for j in range(num_networks):
                if weight_zero[i, j] == 0:
                    continue

                plot_g_vs_z(
                    feats_in_test,
                    feats_out_test,
                    g_sr=None,
                    out_idx=i,
                    feat_idx=j,
                    prime=False,
                    plot_dir=plot_dir
                )
