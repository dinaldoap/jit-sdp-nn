import pandas as pd


def best_configs():
    df_tuning = pd.read_csv('data/tuning.csv')
    df_tuning = df_tuning.dropna(subset=['params.dataset'])
    print(len(df_tuning))
    print(df_tuning.columns)
    print(df_tuning.head())
    '''
    df_groups = df_runs.groupby(by=['params.dataset', 'params.model']).agg({
        'metrics.avg_gmean': ['mean', 'std'], 'metrics.avg_r0-r1': ['mean', 'std']})
    print(df_groups)

    df_tops = df_groups.reset_index()
    df_tops = df_tops.sort_values(
        by=[('metrics.avg_gmean', 'mean')], ascending=False)
    df_tops = df_tops.groupby(by=['params.dataset']).agg(
        {'params.model': ['first']})
    print(df_tops)
    '''


if __name__ == '__main__':
    best_configs()
