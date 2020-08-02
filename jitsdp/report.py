# coding=utf-8
import pandas as pd

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
