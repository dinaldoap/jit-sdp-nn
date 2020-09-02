# coding=utf-8
from jitsdp.plot import plot_recalls_gmean, plot_proportions
from jitsdp.data import load_results
from jitsdp.utils import unique_dir

import pandas as pd
import mlflow


def report(config):
    dir = unique_dir(config)
    results = load_results(dir=dir)
    plot_recalls_gmean(results, config=config, dir=dir)
    plot_proportions(results, config=config, dir=dir)
    metrics = ['r0', 'r1', 'r0-r1', 'gmean', 't1', 's1', 'p1']
    metrics = {'avg_{}'.format(
        metric): results[metric].mean() for metric in metrics}
    mlflow.log_metrics(metrics)
    mlflow.log_artifacts(local_dir=dir)


def print_results():
    df_runs = pd.read_csv('data/runs.csv')
    df_runs = df_runs.dropna(subset=['params.dataset'])
    df_groups = df_runs.groupby(by=['params.dataset', 'params.model']).agg({
        'metrics.avg_gmean': ['mean', 'std'], 'metrics.avg_r0-r1': ['mean', 'std']})
    print(df_groups)

    df_tops = df_groups.reset_index()
    df_tops = df_tops.sort_values(
        by=[('metrics.avg_gmean', 'mean')], ascending=False)
    df_tops = df_tops.groupby(by=['params.dataset']).agg(
        {'params.model': ['first']})
    print(df_tops)


if __name__ == '__main__':
    print_results()
