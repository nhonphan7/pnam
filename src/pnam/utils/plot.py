import glob
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from tbparse import SummaryReader


def plot_vars(
    X: np.ndarray,
    y: np.ndarray,
    X_labels: list = None,
    y_labels: list = None,
    plot_dir: str = None
) -> None:
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=-1)
    num_inputs = X.shape[1]
    num_targets = y.shape[1]
    for i in range(num_targets):
        for j in range(num_inputs):
            plt.figure(figsize=(8, 6))
            plt.plot(X[:, j], y[:, i], 'k.', markersize=10.)
            if X_labels and y_labels:
                plt.xlabel(X_labels[j])
                plt.ylabel(y_labels[i])
            plt.grid()
            plt.tight_layout()
            if plot_dir:
                plt.savefig(f'{plot_dir}/{y_labels[i]}_vs_{X_labels[j]}.png')
                plt.close()
            else:
                plt.show()


def plot_loss_or_metric(
    time: str,
    model_idx: int = 0,
    loss_or_metric: str = 'loss',
    semilogy: bool = False,
    plot_dir: str = None
) -> None:
    list_of_files = glob.glob(f'{time}/pnam/{model_idx}/logs/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    df = SummaryReader(latest_file).scalars
    loss_or_metric = loss_or_metric.lower()
    if loss_or_metric == 'acc':
        y_label = 'Accuracy'
    elif loss_or_metric in ['auroc', 'ap', 'mse', 'mae']:
        y_label = loss_or_metric.upper()
    else:
        y_label = loss_or_metric.title()
    loss_or_metric = loss_or_metric.title()
    df_train = df[df['tag'] == f'Logs/{loss_or_metric}TrainEpoch']
    df_train = df_train.drop('tag', axis=1).to_numpy()
    df_val = df[df['tag'] == f'Logs/{loss_or_metric}ValEpoch']
    df_val = df_val.drop('tag', axis=1).to_numpy()
    plt.figure(figsize=(8, 6))
    if semilogy:
        plt.semilogy(
            df_train[:, 0], df_train[:, 1], '-r', label='Train', linewidth=3.
        )
        plt.semilogy(
            df_val[:, 0], df_val[:, 1], '--b', label='Val', linewidth=3.
        )
    else:
        plt.plot(
            df_train[:, 0], df_train[:, 1], '-r', label='Train', linewidth=3.
        )
        plt.plot(df_val[:, 0], df_val[:, 1], '--b', label='Val', linewidth=3.)
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.legend(loc='best')
    plt.grid()
    plt.tight_layout()
    if plot_dir:
        plt.savefig(f'{plot_dir}/{loss_or_metric.lower()}.png')
        plt.close()
    else:
        plt.show()


def plot_weight(
    weight: np.ndarray,
    color: str = None,
    labels: list = None,
    colorbar: bool = True,
    plot_dir: str = None,
    title: str = None
) -> None:
    min_weight = weight.min()
    max_weight = weight.max()
    if min_weight >= 0:
        colormap = mpl.colormaps['Reds'].resampled(256)
        norm = colors.Normalize(vmin=0., vmax=max_weight)
    elif max_weight <= 0:
        colormap = mpl.colormaps['Blues'].resampled(256)
        norm = colors.Normalize(vmin=min_weight, vmax=0.)
    else:
        colormap = mpl.colormaps['RdBu_r'].resampled(256)
        norm = colors.TwoSlopeNorm(vcenter=0., vmin=min_weight, vmax=max_weight)
    if color:
        colormap = mpl.colormaps[color].resampled(256)
    if max_weight <= 0:
        cmap = colors.ListedColormap(colormap(np.linspace(1., 0., 256)))
    else:
        cmap = colors.ListedColormap(colormap(np.linspace(0., 1., 256)))
    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.imshow(weight, cmap=cmap, norm=norm)
    ax.grid(which='major', axis='both', color='k', linestyle='-', linewidth=2.)
    if labels:
        ax.set_xticks(
            np.arange(-0.5, weight.shape[1], 1), labels + [''], rotation=45.
        )
    else:
        ax.set_xticks(np.arange(-0.5, weight.shape[1], 1))
    ax.set_yticks(np.arange(-0.5, weight.shape[0], 1))
    if labels:
        ax.tick_params(bottom=False, left=False, labelleft=False)
    else:
        ax.tick_params(
            bottom=False, left=False, labelbottom=False, labelleft=False
        )
    if colorbar:
        fig.colorbar(img, ax=ax)
    plt.tight_layout()
    if plot_dir and title:
        plt.savefig(f'{plot_dir}/{title}.png')
        plt.close()
    else:
        plt.show()


def plot_mean(
    mean: np.ndarray, labels: list, plot_dir: str = None, title: str = None
) -> None:
    # Copy labels to avoid modifying caller's list
    labels = list(labels)
    sort_indices = np.argsort(mean)[::-1]
    for i in range(mean.shape[0]):
        labels[sort_indices[i]] = str(i + 1) + '. ' + labels[sort_indices[i]]
    plt.figure(figsize=(8, 6))
    plt.barh(labels[::-1], mean[::-1], color='r')
    plt.xlabel('Mean absolute coefficient')
    plt.grid()
    plt.tight_layout()
    if plot_dir and title:
        plt.savefig(f'{plot_dir}/{title}.png')
        plt.close()
    else:
        plt.show()


def plot_top_n_acc(
    acc: np.ndarray, graph: str = 'bar', plot_dir: str = None
) -> None:
    top_n = acc.shape[0] - np.arange(acc.shape[0])
    plt.figure(figsize=(8, 6))
    if graph == 'bar':
        plt.bar(top_n, acc, color='r')
    elif graph == 'line':
        plt.plot(top_n, acc, '-r', linewidth=3.)
    plt.xlabel('Top $n$ inputs')
    plt.ylabel('Accuracy')
    if graph == 'bar':
        plt.xticks(top_n)
    plt.yticks(np.linspace(0., 1., 11))
    plt.gca().invert_xaxis()
    plt.grid()
    plt.tight_layout()
    if plot_dir:
        plt.savefig(f'{plot_dir}/top_n_acc.png')
        plt.close()
    else:
        plt.show()


def plot_g_vs_z(
    z: np.ndarray,
    g_pnam: np.ndarray,
    g_sr: np.ndarray = None,
    out_idx: int = 0,
    feat_idx: int = 0,
    prime: bool = False,
    plot_dir: str = None
) -> None:
    sort_indices = np.argsort(z[:, feat_idx])
    out_comp = out_idx + 1
    feat_comp = feat_idx + 1
    plt.figure(figsize=(8, 6))
    plt.plot(
        z[sort_indices, feat_idx],
        g_pnam[sort_indices, out_idx, feat_idx],
        '-r',
        linewidth=3.
    )
    if g_sr is not None:
        plt.plot(
            z[sort_indices, feat_idx],
            g_sr[sort_indices, out_idx, feat_idx],
            '--b',
            linewidth=3.
        )
    plt.xlabel('$z_{%s}$' % feat_comp)
    if prime:
        plt.ylabel(r'$g_{%s,%s}^\prime$' % (out_comp, feat_comp))
    else:
        plt.ylabel('$g_{%s,%s}$' % (out_comp, feat_comp))
    if g_sr is not None:
        plt.legend(labels=['PNAM', 'SR'], loc='best')
    plt.grid()
    plt.tight_layout()
    if plot_dir:
        if prime:
            plt.savefig(
                f'{plot_dir}/gp{out_comp}_{feat_comp}_vs_z{feat_comp}.png'
            )
        else:
            plt.savefig(
                f'{plot_dir}/g{out_comp}_{feat_comp}_vs_z{feat_comp}.png'
            )
        plt.close()
    else:
        plt.show()
