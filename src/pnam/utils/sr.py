from typing import Tuple

import os
import pysr
import sympy
import numpy as np
import pandas as pd


def fit(
    X: np.ndarray,
    y: np.ndarray,
    yp: np.ndarray,
    feat_comp: int,
    model_sr: pysr.PySRRegressor
) -> Tuple[str, sympy.Function, sympy.Function, sympy.Function]:
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis=-1)

    model_sr.fit(X, y, weights=yp, variable_names=[f'z_{feat_comp}'])
    g_latex = model_sr.latex(precision=3)
    g_sympy = model_sr.sympy()

    z = sympy.symbols(f'z_{feat_comp}')
    gp_sympy = sympy.diff(g_sympy, z)
    gpp_sympy = sympy.diff(gp_sympy, z)
    return g_latex, g_sympy, gp_sympy, gpp_sympy


def predict(
    X: np.ndarray,
    y: np.ndarray,
    yp: np.ndarray,
    feat_comp: int,
    g_sympy: sympy.Function
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    z = sympy.symbols(f'z_{feat_comp}')
    gp_sympy = sympy.diff(g_sympy, z)
    gpp_sympy = sympy.diff(gp_sympy, z)

    g_lamb = sympy.lambdify(z, g_sympy)
    gp_lamb = sympy.lambdify(z, gp_sympy)
    gpp_lamb = sympy.lambdify(z, gpp_sympy)

    g_pred = g_lamb(X)
    gp_pred = gp_lamb(X)
    gpp_pred = gpp_lamb(X)

    pred_diff = g_pred - y
    if yp is not None:
        grad_diff = gp_pred - yp
        loss = ((pred_diff**2).sum() + (grad_diff**2).sum()) / len(pred_diff)
    else:
        loss = (pred_diff**2).sum() / len(pred_diff)
    return g_pred, gp_pred, gpp_pred, loss


def train_or_evaluate_all(
    time: str,
    sobolev: bool,
    model_idx: int,
    weight: np.ndarray,
    feats_out_train: np.ndarray,
    feats_in_train: np.ndarray,
    grad_in_train: np.ndarray,
    feats_out_test: np.ndarray,
    feats_in_test: np.ndarray,
    grad_in_test: np.ndarray,
    model_sr: pysr.PySRRegressor,
    train: bool = True
) -> Tuple[np.ndarray, ...]:
    data_path = f'{time}/sr/{model_idx}/equations.csv'
    if train or not os.path.exists(data_path):
        data = []
    else:
        df = pd.read_csv(data_path)

    g_sr = np.zeros_like(feats_out_test)
    gp_sr = np.zeros_like(feats_out_test)
    gpp_sr = np.zeros_like(feats_out_test)

    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            if weight[i, j] == 0:
                continue
            out_comp = i + 1
            feat_comp = j + 1

            if train or not os.path.exists(data_path):
                g_latex, g_sympy, gp_sympy, gpp_sympy = fit(
                    feats_in_train[:, j],
                    feats_out_train[:, i, j],
                    grad_in_train[:, i, j] if sobolev else None,
                    feat_comp,
                    model_sr
                )
            else:
                df_row = df[
                    (df['output'] == out_comp) & (df['z'] == feat_comp)
                ]
                g_sympy = df_row['g_sympy'].to_numpy()[0]

            g_pred, gp_pred, gpp_pred, loss = predict(
                feats_in_test[:, j],
                feats_out_test[:, i, j],
                grad_in_test[:, i, j] if sobolev else None,
                feat_comp,
                g_sympy
            )

            g_sr[:, i, j] = g_pred
            gp_sr[:, i, j] = gp_pred
            gpp_sr[:, i, j] = gpp_pred

            if train or not os.path.exists(data_path):
                data.append([
                    out_comp,
                    feat_comp,
                    g_latex,
                    g_sympy,
                    gp_sympy,
                    gpp_sympy,
                    loss
                ])

    if train or not os.path.exists(data_path):
        df = pd.DataFrame(
            data,
            columns=[
                'output',
                'z',
                'g_latex',
                'g_sympy',
                'gp_sympy',
                'gpp_sympy',
                'loss'
            ]
        )
        df.to_csv(data_path, index=False)
    return g_sr, gp_sr, gpp_sr


def retrain_one(
    time: str,
    sobolev: bool,
    model_idx: int,
    feats_out_train: np.ndarray,
    feats_in_train: np.ndarray,
    grad_in_train: np.ndarray,
    feats_out_test: np.ndarray,
    feats_in_test: np.ndarray,
    grad_in_test: np.ndarray,
    g_sr: np.ndarray,
    gp_sr: np.ndarray,
    gpp_sr: np.ndarray,
    out_idx: int,
    feat_idx: int,
    model_sr: pysr.PySRRegressor
) -> Tuple[np.ndarray, ...]:
    out_comp = out_idx + 1
    feat_comp = feat_idx + 1

    g_latex, g_sympy, gp_sympy, gpp_sympy = fit(
        feats_in_train[:, feat_idx],
        feats_out_train[:, out_idx, feat_idx],
        grad_in_train[:, out_idx, feat_idx] if sobolev else None,
        feat_comp,
        model_sr
    )

    g_pred, gp_pred, gpp_pred, loss = predict(
        feats_in_test[:, feat_idx],
        feats_out_test[:, out_idx, feat_idx],
        grad_in_test[:, out_idx, feat_idx] if sobolev else None,
        feat_comp,
        g_sympy
    )

    g_sr[:, out_idx, feat_idx] = g_pred
    gp_sr[:, out_idx, feat_idx] = gp_pred
    gpp_sr[:, out_idx, feat_idx] = gpp_pred

    data_path = f'{time}/sr/{model_idx}/equations.csv'
    df = pd.read_csv(data_path)
    df.loc[(df['output'] == out_comp) & (df['z'] == feat_comp)] = [
        out_comp, feat_comp, g_latex, g_sympy, gp_sympy, gpp_sympy, loss
    ]
    df.to_csv(data_path, index=False)
    return g_sr, gp_sr, gpp_sr


def construct_eq(
    proj_mat: np.ndarray,
    feat_idx: int,
    g_sympy: sympy.Function,
    X_scale_: np.ndarray = None,
    X_min_: np.ndarray = None,
    X_mean_: np.ndarray = None,
    simplify: bool = True
) -> sympy.Function:
    z = ''
    for i in range(proj_mat.shape[1]):
        if X_min_ is not None:
            x = f'(x_{i + 1}*{X_scale_[i]}+{X_min_[i]})'
        elif X_mean_ is not None:
            x = f'(x_{i + 1}-{X_mean_[i]})/{X_scale_[i]}'
        else:
            x = f'x_{i + 1}'
        z += f'+{proj_mat[feat_idx, i]}*{x}'

    g = g_sympy.replace(f'z_{feat_idx + 1}', f'({z})')
    g = sympy.sympify(g)
    if simplify:
        g = sympy.simplify(g)
    g = g.replace(lambda x: isinstance(x, sympy.Float), lambda x: x.round(3))
    return g


def combine_eqs(
    time: str,
    model_idx: int,
    weight: np.ndarray,
    bias: np.ndarray,
    proj_mat: np.ndarray,
    out_idx: int,
    X_scale_: np.ndarray = None,
    X_min_: np.ndarray = None,
    X_mean_: np.ndarray = None,
    y_scale_: np.ndarray = None,
    y_min_: np.ndarray = None,
    y_mean_: np.ndarray = None,
    simplify: bool = True
) -> sympy.Function:
    df = pd.read_csv(f'{time}/sr/{model_idx}/equations.csv')

    f = ''
    for i in range(proj_mat.shape[0]):
        if weight[out_idx, i] == 0:
            continue
        df_row = df[(df['output'] == out_idx + 1) & (df['z'] == i + 1)]
        g_sympy = df_row['g_sympy'].to_numpy()[0]
        g = construct_eq(
            proj_mat, i, g_sympy, X_scale_, X_min_, X_mean_, simplify
        )
        f += f'+{g}'
    f += f'+{bias[out_idx]}'

    if y_min_ is not None:
        f = f'({f}-{y_min_[out_idx]})/{y_scale_[out_idx]}'
    elif y_mean_ is not None:
        f = f'({f})*{y_scale_[out_idx]}+{y_mean_[out_idx]}'

    f = sympy.sympify(f)
    if simplify:
        f = sympy.simplify(f)
    f = f.replace(lambda x: isinstance(x, sympy.Float), lambda x: x.round(3))
    return f


def evaluate_eq(
    X: np.ndarray,
    weight: np.ndarray,
    proj_mat: np.ndarray,
    out_idx: int,
    eq: sympy.Function
) -> Tuple[str, np.ndarray]:
    eq_latex = sympy.latex(eq)

    feat_indices = []
    for i in range(proj_mat.shape[0]):
        if weight[out_idx, i] == 0:
            continue
        feat_indices.append(i)

    proj_mat_mean = np.mean(abs(proj_mat[feat_indices]), axis=0)

    input_indices = []
    for i in range(proj_mat_mean.shape[0]):
        if proj_mat_mean[i] == 0:
            continue
        input_indices.append(i)

    eq_lamb = sympy.lambdify(
        sympy.symbols([[f'x_{i + 1}' for i in input_indices]]), eq
    )
    eq_pred = eq_lamb([X[:, i] for i in input_indices])
    return eq_latex, eq_pred
