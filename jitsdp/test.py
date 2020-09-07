import numpy as np
import pandas as pd


def best_configs():
    df_tuning = pd.read_csv('data/tuning.csv')
    #assert np.all(df_tuning[df_tuning['status'] == 'FINISHED'])
    print(df_tuning.columns)
    print(df_tuning.head(1))
    config_cols = config_columns(df_tuning.columns)
    df_best_configs = df_tuning.groupby(by=config_cols, as_index=False).agg({
        'metrics.avg_gmean': 'mean', 'tags.run.command': 'first'})
    df_best_configs.columns = remove_columns_prefix(df_best_configs.columns)
    df_best_configs.sort_values(by='avg_gmean', ascending=False)
    df_best_configs.drop_duplicates(subset=['meta_model', 'model', 'dataset'])
    print(len(df_best_configs))
    print(df_best_configs.columns)
    print(df_best_configs.head(1))


def config_columns(cols):
    return [col for col in cols if col.startswith('params') and not col.endswith('seed')]


def remove_columns_prefix(cols):
    return ['.'.join(col.split('.')[1:]) for col in cols]


if __name__ == '__main__':
    best_configs()
