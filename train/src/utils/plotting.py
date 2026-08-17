import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import numpy as np
from tslearn.metrics import dtw as ts_dtw
from statsmodels.tsa.stattools import acf
from src.exp.exp_joint_kl_div import Exp_Joint_KL_Div
from src.utils.metrcis import riemannian_distance
from scipy.stats import entropy
from src.utils.utils import dotdict

sns.set_style('whitegrid')


def plot_epoch_loss(epoch_loss, title=None, save_name=None, log_scale=False):

    epochs = range(1, len(epoch_loss) + 1)
    if log_scale:
        loss_history = np.log(epoch_loss)
    else:
        loss_history = epoch_loss

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_history, 'b-o')
    plt.xlabel('Step', fontsize=14)
    if log_scale:
        plt.ylabel('Log Loss', fontsize=14)
    else:
        plt.ylabel('Loss', fontsize=14)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.clf()


def plot_gradnorm_weights(gradnorm_wrapper, save_name=None):
    weights_history = np.array(gradnorm_wrapper.weights_history)  # Shape: (num_steps, num_tasks)
    num_tasks = weights_history.shape[1]

    plt.figure(figsize=(12, 5))
    for i in range(num_tasks):
        plt.plot(weights_history[:, i], label=f'{gradnorm_wrapper.loss_object.name[i]} Weight')
    plt.xlabel("Step")
    plt.ylabel("Weight Magnitude")
    plt.title("Adaptive Weights during Training", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)


def plot_generated_vs_benchmark_dist(benchmark, output, save_name=None, bins=100, joint_kl=False, configs=None):
    col_labels = benchmark.columns[1:]
    main_name = save_name[:-4] if save_name is not None else None

    num_plots = len(col_labels)
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 8, num_rows * 6))
    axes = axes.flatten()

    m_kl_div = marginal_kl_divergence(output.values, benchmark.iloc[:, 1:].values, bins=bins)
    if joint_kl:
        j_kl_div = joint_kl_divergence(configs, output.values, benchmark.iloc[:, 1:].values)
    else:
        j_kl_div = -1

    for i in range(num_plots):
        sns.histplot(output.iloc[:, i], bins=bins, color='red', label='Generated', kde=True, stat="density", alpha=0.5,
                     ax=axes[i])
        sns.histplot(benchmark.iloc[:, i + 1], bins=bins, color='blue', label='Benchmark', kde=True, stat="density",
                     alpha=0.3, ax=axes[i])

        axes[i].set_title(f'Distribution of {col_labels[i]}')
        axes[i].set_xlabel('Values')
        axes[i].set_ylabel('Density')
        axes[i].legend()

    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if save_name is not None:
        plt.savefig(main_name + '_plots.png', dpi=300)
    else:
        plt.show()
    plt.clf()

    return m_kl_div, j_kl_div


def plot_generated_vs_benchmark_moments(benchmark, output, save_name=None):
    out_mean, out_std, out_s, out_kurt = get_moments(output)
    benchmark_mean, benchmark_std, benchmark_s, benchmark_kurt = get_moments(benchmark.iloc[:, 1:].values)

    x = np.arange(len(out_mean))
    x_labels = benchmark.columns[1:]

    def format_mse(mse):
        return f'{mse:.2e}' if mse < 0.01 else f'{mse:.2f}'

    plt.figure(figsize=(15, 10))
    width = 0.35

    plt.subplot(2, 2, 1)
    mse_mean = np.mean((out_mean - benchmark_mean) ** 2)
    plt.bar(x - width / 2, out_mean, width, label='Generated')
    plt.bar(x + width / 2, benchmark_mean, width, label='Benchmark')
    plt.title(f'Mean - MSE: {format_mse(mse_mean)}')
    plt.xticks(ticks=x, labels=x_labels, rotation=45)
    plt.legend()

    plt.subplot(2, 2, 2)
    mse_std = np.mean((out_std - benchmark_std) ** 2)
    plt.bar(x - width / 2, out_std, width, label='Generated')
    plt.bar(x + width / 2, benchmark_std, width, label='Benchmark')
    plt.title(f'Standard Deviation - MSE: {format_mse(mse_std)}')
    plt.xticks(ticks=x, labels=x_labels, rotation=45)
    plt.legend()

    plt.subplot(2, 2, 3)
    mse_s = np.mean((out_s - benchmark_s) ** 2)
    plt.bar(x - width / 2, out_s, width, label='Generated')
    plt.bar(x + width / 2, benchmark_s, width, label='Benchmark')
    plt.title(f'Skewness - MSE: {format_mse(mse_s)}')
    plt.xticks(ticks=x, labels=x_labels, rotation=45)
    plt.legend()

    plt.subplot(2, 2, 4)
    mse_kurt = np.mean((out_kurt - benchmark_kurt) ** 2)
    plt.bar(x - width / 2, out_kurt, width, label='Generated')
    plt.bar(x + width / 2, benchmark_kurt, width, label='Benchmark')
    plt.title(f'Kurtosis - MSE: {format_mse(mse_kurt)}')
    plt.xticks(ticks=x, labels=x_labels, rotation=45)
    plt.legend()

    plt.suptitle('Generated vs Benchmark Moments')
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name)
    else:
        plt.show()
    plt.clf()

    return mse_mean, mse_std, mse_s, mse_kurt


def plot_generated_vs_benchmark_autocorr(benchmark, output, save_name=None, absolute=True, lags=20):
    col_labels = benchmark.columns[1:]
    main_name = save_name[:-4] if save_name is not None else None
    dtw_distances = []

    num_plots = len(col_labels)
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 8, num_rows * 6))
    axes = axes.flatten()

    for i in range(num_plots):
        output_ts = abs(output.iloc[:, i]) if absolute else output.iloc[:, i]
        benchmark_ts = abs(benchmark.iloc[:, i + 1]) if absolute else benchmark.iloc[:, i + 1]

        acf_output = acf(output_ts, nlags=lags)[1:]
        acf_output = np.nan_to_num(acf_output, nan=0.0)  # Replace NaNs with 0
        acf_benchmark = acf(benchmark_ts, nlags=lags)[1:]
        distance = ts_dtw(acf_output, acf_benchmark)
        dtw_distances.append(distance)

        stem_output = axes[i].stem(range(1, len(acf_output) + 1), acf_output, linefmt='r-', markerfmt='ro', basefmt=" ",
                                   label='Generated')
        plt.setp(stem_output[0], alpha=0.7)
        plt.setp(stem_output[1], alpha=0.9)

        stem_benchmark = axes[i].stem(range(1, len(acf_benchmark) + 1), acf_benchmark, linefmt='b-', markerfmt='bo',
                                      basefmt=" ", label='Benchmark')
        plt.setp(stem_benchmark[0], alpha=0.7)
        plt.setp(stem_benchmark[1], alpha=0.9)

        axes[i].set_title(f'ACF of {col_labels[i]} - DTW Dist: {distance:.2f}')
        axes[i].set_xlabel('Lags')
        axes[i].set_ylabel('Autocorrelation')
        axes[i].legend()

    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if save_name is not None:
        plt.savefig(main_name + '_plots.png', dpi=300)
    else:
        plt.show()
    plt.clf()

    return np.mean(dtw_distances)


def plot_generated_vs_benchmark_cov(benchmark, output, save_name=None):
    main_name = save_name[:-4] if save_name is not None else None

    output_cov = output.cov().values
    benchmark_cov = benchmark.iloc[:, 1:].cov().values
    col_labels = benchmark.columns[1:]

    cov_diff = output_cov - benchmark_cov

    global_min = min(np.min(output_cov), np.min(benchmark_cov), np.min(cov_diff))
    global_max = max(np.max(output_cov), np.max(benchmark_cov), np.max(cov_diff))

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), gridspec_kw={'width_ratios': [1, 1, 1]})

    sns.heatmap(output_cov, annot=True, cmap='coolwarm', center=0, vmin=global_min, vmax=global_max,
                xticklabels=col_labels, yticklabels=col_labels, ax=axes[0], cbar=False)
    axes[0].set_title('Generated Covariance')
    axes[0].set_xlabel('Assets')
    axes[0].set_ylabel('Assets')
    axes[0].invert_yaxis()

    sns.heatmap(benchmark_cov, annot=True, cmap='coolwarm', center=0, vmin=global_min, vmax=global_max,
                xticklabels=col_labels, yticklabels=col_labels, ax=axes[1], cbar=False)
    axes[1].set_title('Benchmark Covariance')
    axes[1].set_xlabel('Assets')
    axes[1].set_ylabel('Assets')
    axes[1].invert_yaxis()

    sns.heatmap(cov_diff, annot=True, cmap='coolwarm', center=0, vmin=global_min, vmax=global_max,
                xticklabels=col_labels, yticklabels=col_labels, ax=axes[2], cbar=True,
                cbar_ax=fig.add_axes([0.93, 0.3, 0.02, 0.4]))
    axes[2].set_title('Difference between Generated and Benchmark Covariance')
    axes[2].set_xlabel('Assets')
    axes[2].set_ylabel('Assets')
    axes[2].invert_yaxis()

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if save_name is not None:
        plt.savefig(main_name + '_plots.png')
    else:
        plt.show()
    plt.clf()

    # mse_cov = np.mean((np.triu(output_cov) - np.triu(benchmark_cov)) ** 2)

    try:
        r_dist = riemannian_distance(output_cov, benchmark_cov)
    except ValueError as e:
        print(f"Error calculating Riemannian distance: {e}")
        r_dist = 99

    return output_cov, benchmark_cov, cov_diff, r_dist


def plot_generated_vs_benchmark_corr(benchmark, output, save_name=None):
    main_name = save_name[:-4] if save_name is not None else None

    output_corr = output.corr().values
    benchmark_corr = benchmark.iloc[:, 1:].corr().values
    col_labels = benchmark.columns[1:]

    corr_diff = output_corr - benchmark_corr

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), gridspec_kw={'width_ratios': [1, 1, 1]})

    sns.heatmap(output_corr, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                xticklabels=col_labels, yticklabels=col_labels, ax=axes[0], cbar=False)
    axes[0].set_title('Generated Correlation')
    axes[0].set_xlabel('Assets')
    axes[0].set_ylabel('Assets')
    axes[0].invert_yaxis()

    sns.heatmap(benchmark_corr, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                xticklabels=col_labels, yticklabels=col_labels, ax=axes[1], cbar=False)
    axes[1].set_title('Benchmark Correlation')
    axes[1].set_xlabel('Assets')
    axes[1].set_ylabel('Assets')
    axes[1].invert_yaxis()

    sns.heatmap(corr_diff, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                xticklabels=col_labels, yticklabels=col_labels, ax=axes[2], cbar=True,
                cbar_ax=fig.add_axes([0.93, 0.3, 0.02, 0.4]))
    axes[2].set_title('Difference between Generated and Benchmark Correlation')
    axes[2].set_xlabel('Assets')
    axes[2].set_ylabel('Assets')
    axes[2].invert_yaxis()

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if save_name is not None:
        plt.savefig(main_name + '_plots.png')
    else:
        plt.show()
    plt.clf()

    try:
        r_dist = riemannian_distance(output_corr, benchmark_corr)
    except ValueError as e:
        print(f"Error calculating Riemannian distance: {e}")
        r_dist = 99

    return output_corr, benchmark_corr, corr_diff, r_dist


def get_moments(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    s = skew(data, axis=0)
    kurt = kurtosis(data, axis=0) + 3
    return mean, std, s, kurt


def plot_results_dict(results_dict, save_name):
    steps = results_dict['Step']
    metrics = {key: value for key, value in results_dict.items() if key != 'Step'}

    num_plots = len(metrics)
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 8, num_rows * 6))
    axes = axes.flatten()

    for idx, (metric_name, metric_values) in enumerate(metrics.items()):
        axes[idx].plot(steps, metric_values, marker='o', linestyle='-', color='blue', alpha=0.7)
        axes[idx].set_title(f'{metric_name}')
        axes[idx].set_xlabel('Step')
        axes[idx].set_ylabel(metric_name)
        axes[idx].grid(True)

    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.clf()


def marginal_kl_divergence(arr1, arr2, bins=50):
    kl_results = []
    epsilon = 1e-10

    for i in range(arr1.shape[1]):
        p_hist, bin_edges = np.histogram(arr1[:, i], bins=bins, density=True)
        q_hist, _ = np.histogram(arr2[:, i], bins=bin_edges, density=True)

        p_hist = p_hist + epsilon
        q_hist = q_hist + epsilon

        p_hist = p_hist / p_hist.sum()
        q_hist = q_hist / q_hist.sum()

        kl = entropy(p_hist, q_hist)
        kl_results.append(kl)

    mean_kl = np.mean(kl_results)
    return mean_kl


def joint_kl_divergence(configs, X, Y):
    assert X.shape[1] == Y.shape[1], 'X and Y must have the same number of columns'

    args = {
        'checkpoints': configs.checkpoints,
        'num_workers': configs.num_workers,
        'use_gpu': configs.use_gpu,
        'gpu': configs.gpu,
        'compile': configs.compile,
        'num_samples': 10000,
        'batch_size': 1024,
        'train_epochs': 50,
        'learning_rate': 0.0001,
        'ma_rate': 0.1,
        'input_size': X.shape[1],
        'hidden_size': 512,
        'ma_ef': 1,
    }
    args = dotdict(args)
    exp = Exp_Joint_KL_Div(args, X, Y)
    KL_div = exp.train()

    return KL_div