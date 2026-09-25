import datetime
import joblib
import os
import pysr
import sympy
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sys.path.append('../')
from pnam.utils.plot import *
from pnam.utils.sr import *
from pnam.wrapper import PNAMBase


def plot_preds(X, y, preds, X_label, y_label, plot_dir=None, title=None):
    if X_label == 'Strain':
        input = X[:, 0]
        input_label = 'Strain'
    elif X_label == 'Xi':
        input = X[:, 1]
        input_label = 'Order parameter'
    if y_label == 'W':
        output = y[:, 0]
        predictions = preds[:, 0]
        output_label = 'Strain energy [MPa]'
    elif y_label == 'f':
        output = y[:, 1]
        predictions = preds[:, 1]
        output_label = 'Phase energy [MPa]'
    elif y_label == 'Stress':
        output = y[:, 2] / 1000
        predictions = preds[:, 2] / 1000
        output_label = 'Stress [GPa]'
    sort_indices = np.argsort(input)
    plt.figure(figsize=(8, 6))
    plt.plot(
        input[sort_indices],
        output[sort_indices],
        'k.',
        label='True',
        markersize=10.
    )
    plt.plot(
        input[sort_indices],
        predictions[sort_indices],
        '-r',
        label='Pred',
        linewidth=3.
    )
    plt.xlabel(input_label)
    plt.ylabel(output_label)
    plt.legend(loc='best')
    plt.grid()
    plt.tight_layout()
    if plot_dir:
        if title:
            plt.savefig(f'{plot_dir}/{y_label}_vs_{X_label}_{title}.png')
        else:
            plt.savefig(f'{plot_dir}/{y_label}_vs_{X_label}.png')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    time = datetime.datetime.now().isoformat()
    # time = 'phase_field_pnam8'

    small_size = 20
    large_size = 24
    plt.rc('font', size=small_size)
    plt.rc('axes', labelsize=large_size)

    data_path = '../../data/phase_field.csv'
    df = pd.read_csv(data_path)
    
    X_labels = ['Strain', 'Xi', 'Nabla Xi']
    y_labels = ['W', 'f', 'Stress']

    # plot_vars(
    #     df[X_labels].to_numpy(),
    #     df[y_labels].to_numpy(),
    #     X_labels=X_labels,
    #     y_labels=y_labels
    # )
    
    X_scaler = MinMaxScaler()
    df[X_labels] = X_scaler.fit_transform(df[X_labels])
    y_scaler = MinMaxScaler()
    df[y_labels] = y_scaler.fit_transform(df[y_labels])

    random_state = 42
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=random_state
    )
    X_train, X_test = df_train[X_labels].to_numpy(), df_test[X_labels].to_numpy()
    y_train, y_test = df_train[y_labels].to_numpy(), df_test[y_labels].to_numpy()

    pnam = True
    num_inputs = X_train.shape[1]
    proj_size = 8
    num_networks = proj_size if proj_size > 0 else num_inputs
    num_outputs = 2
    metric = None
    scale = True
    sobolev = True
    
    model_idx = 0
    weight_thresh = 0.05
    proj_mat_thresh = 0.05

    sr = True
    num_sr_points = 1000

    if scale and sobolev:
        scaler_dir = f'{time}/pnam/scaler'
        os.makedirs(scaler_dir, exist_ok=True)
        joblib.dump(X_scaler, f'{scaler_dir}/X_scaler.pkl')
        joblib.dump(y_scaler, f'{scaler_dir}/y_scaler.pkl')

    model_pnam = PNAMBase(
        random_state=random_state,
        num_learners=10,
        pnam=pnam,
        std=0.,
        proj_size=proj_size,
        hidden_sizes=[256, 256, 256],
        num_outputs=num_outputs,
        activation=F.silu,
        dropout=0.,
        feature_dropout=0.,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        regression=True,
        rot_reg=0.001,
        proj_reg=0.001,
        weight_reg=0.001,
        output_reg=0.001,
        l2_reg=0.001,
        verbose=False,
        metric=metric,
        scale=scale,
        sobolev=sobolev,
        val_split=0.2,
        n_jobs=4,
        batch_size=256,
        log_dir=f'{time}/pnam',
        lr=0.001,
        decay_step=1,
        decay_rate=0.995,
        num_epochs=5000,
        energy=True,
        patience=50
    )
    
    model_pnam.fit(X_train, y_train)
    # model_pnam.load_checkpoints(f'{time}/pnam')

    plot_dir = f'{time}/figures/{model_idx}'
    os.makedirs(plot_dir, exist_ok=True)
    print(f'Model index: {model_idx}')

    plot_loss_or_metric(
        time, model_idx, 'loss', semilogy=True, plot_dir=plot_dir
    )
    if metric:
        plot_loss_or_metric(
            time, model_idx, metric, semilogy=True, plot_dir=plot_dir
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

    X = X_scaler.inverse_transform(X_test)
    y = y_scaler.inverse_transform(y_test)
    preds = y_scaler.inverse_transform(preds_test)
    plot_preds(X, y, preds, X_label='Strain', y_label='W', plot_dir=plot_dir)
    plot_preds(X, y, preds, X_label='Xi', y_label='f', plot_dir=plot_dir)
    plot_preds(
        X, y, preds, X_label='Strain', y_label='Stress', plot_dir=plot_dir
    )

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
            
            proj_mat_zero[abs(proj_mat_zero) < proj_mat_thresh] = 0.
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

        X = X_scaler.inverse_transform(X_test)
        y = y_scaler.inverse_transform(y_test)
        preds = y_scaler.inverse_transform(preds_test)
        plot_preds(
            X,
            y,
            preds,
            X_label='Strain',
            y_label='W',
            plot_dir=plot_dir,
            title='zero'
        )
        plot_preds(
            X,
            y,
            preds,
            X_label='Xi',
            y_label='f',
            plot_dir=plot_dir,
            title='zero'
        )
        plot_preds(
            X,
            y,
            preds,
            X_label='Strain',
            y_label='Stress',
            plot_dir=plot_dir,
            title='zero'
        )

        if sr:
            num_train_points = X_train.shape[0]
            if num_train_points > num_sr_points:
                np.random.seed(random_state)
                sr_indices = np.random.choice(
                    num_train_points, num_sr_points, replace=False
                )
                X_sr = X_train[sr_indices]
                y_sr = y_train[sr_indices]
            else:
                X_sr = X_train
                y_sr = y_train

            (
                preds_sr, feats_out_sr, feats_in_sr, _, _, _, grad_in_sr
            ) = model_pnam.predict(
                X_sr,
                y_sr,
                model_idx,
                weight_zero,
                proj_mat_zero if proj_size > 0 else None
            )

            if sobolev:
                pysr.jl.seval('''
                import Pkg
                Pkg.add("Zygote")
                ''')
                pysr.jl.seval('import Zygote')

                objective = '''
                function my_custom_objective(
                    tree, dataset::Dataset{T, L}, options
                )::L where {T, L}
                    pred, grad, flag = eval_diff_tree_array(
                        tree, dataset.X, options, 1
                    )
                    !flag && return L(Inf)
                    pred_diff = pred .- dataset.y
                    grad_diff = grad .- dataset.weights
                    return (
                        sum(pred_diff.^2) + sum(grad_diff.^2)
                    ) / length(pred_diff)
                end
                '''

            model_sr = pysr.PySRRegressor(
                binary_operators=['+', '-', '*'],
                unary_operators=['square'],
                maxsize=30,
                niterations=10000000,
                populations=30,
                population_size=30,
                ncycles_per_iteration=500,
                elementwise_loss=None if sobolev else 'L2DistLoss()',
                loss_function=objective if sobolev else None,
                model_selection='best',
                timeout_in_seconds=60.,
                output_directory=f'{time}/sr/{model_idx}'
            )

            g_sr, gp_sr, gpp_sr = train_or_evaluate_all(
                time,
                sobolev,
                model_idx,
                weight_zero,
                feats_out_sr,
                feats_in_sr,
                grad_in_sr,
                feats_out_test,
                feats_in_test,
                grad_in_test,
                model_sr,
                train=False
            )

            # out_idx, feat_idx = 0, 0
            # g_sr, gp_sr, gpp_sr = retrain_one(
            #     time,
            #     sobolev,
            #     model_idx,
            #     feats_out_sr,
            #     feats_in_sr,
            #     grad_in_sr,
            #     feats_out_test,
            #     feats_in_test,
            #     grad_in_test,
            #     g_sr,
            #     gp_sr,
            #     gpp_sr,
            #     out_idx,
            #     feat_idx,
            #     model_sr
            # )

        if sr:
            eqs = pd.read_csv(f'{time}/sr/{model_idx}/equations.csv')
            
        for i in range(num_outputs):
            for j in range(num_networks):
                if weight_zero[i, j] == 0:
                    continue
                    
                if sr:
                    eq = eqs[(eqs['output'] == i + 1) & (eqs['z'] == j + 1)]
                    g_feat = eq['g_sympy'].to_numpy()[0]
                    g_input = construct_eq(
                        proj_mat_zero if proj_size > 0 else np.eye(num_inputs),
                        feat_idx=j,
                        g_sympy=g_feat,
                        X_scale_=X_scaler.scale_,
                        X_min_=X_scaler.min_
                    )
                    
                    g_feat = sympy.simplify(sympy.sympify(g_feat))
                    g_feat = g_feat.replace(
                        lambda x: isinstance(x, sympy.Float),
                        lambda x: x.round(3)
                    )

                    g_feat = sympy.latex(g_feat)
                    g_input = sympy.latex(g_input)
                    print(f'output: {i + 1}, z: {j + 1}')
                    print('g(z):')
                    print(g_feat)
                    print('g(x):')
                    print(g_input)
                    
                plot_g_vs_z(
                    feats_in_test,
                    feats_out_test,
                    g_sr=g_sr if sr else None,
                    out_idx=i,
                    feat_idx=j,
                    prime=False,
                    plot_dir=plot_dir
                )
                if sobolev:
                    plot_g_vs_z(
                        feats_in_test,
                        grad_in_test,
                        g_sr=gp_sr if sr else None,
                        out_idx=i,
                        feat_idx=j,
                        prime=True,
                        plot_dir=plot_dir
                    )

        if sr:
            preds = np.zeros_like(y_test)
            preds[:, :-1] = (
                np.sum(g_sr, axis=-1) + bias - y_scaler.min_[:-1]
            ) / y_scaler.scale_[:-1]
            jac = gp_sr @ proj_mat_zero if proj_size > 0 else gp_sr
            preds_grad = jac[:, :, 0] * X_scaler.scale_[0] / y_scaler.scale_[:-1]
            preds[:, 2] = np.sum(preds_grad, axis=1)
            loss = sk_metrics.mean_squared_error(
                y_test, preds * y_scaler.scale_ + y_scaler.min_
            )
            print(f'Loss: {loss:.10f}')

            X = X_scaler.inverse_transform(X_test)
            y = y_scaler.inverse_transform(y_test)
            plot_preds(
                X,
                y,
                preds,
                X_label='Strain',
                y_label='W',
                plot_dir=plot_dir,
                title='sr'
            )
            plot_preds(
                X,
                y,
                preds,
                X_label='Xi',
                y_label='f',
                plot_dir=plot_dir,
                title='sr'
            )
            plot_preds(
                X,
                y,
                preds,
                X_label='Strain',
                y_label='Stress',
                plot_dir=plot_dir,
                title='sr'
            )
