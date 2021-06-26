from jitsdp.data import DATASETS, load_runs, make_stream
from jitsdp.utils import filename_to_path

import mlflow
import pandas as pd
from pathlib import Path


def add_arguments(parser, filename):
    parser.add_argument('--filename',   type=str,
                        help='Output filename.', default=filename)
    parser.add_argument('--tuning-experiment-name',   type=str,
                        help='Experiment name used for tuning (default: Default).', default='Default')
    parser.add_argument('--testing-experiment-name',   type=str,
                        help='Experiment name used for testing (default: testing).', default='testing')
    parser.add_argument('--format',   type=str, nargs='+', choices=['pickle', 'csv'],
                        help='Format to export (default: pickle]).', default=['pickle'])


def generate(config):
    for dataset in sorted(DATASETS):
        df_dataset = make_stream(dataset)
        export_dataframe(df_dataset, 'datasets/{}'.format(dataset), config)

    for key_experiment_name in ['tuning_experiment_name', 'testing_experiment_name']:
        experiment_name = config[key_experiment_name]
        experiment_id = mlflow.get_experiment_by_name(
            experiment_name).experiment_id
        df_experiment = load_runs(experiment_id)
        export_dataframe(df_experiment, experiment_name, config)
        for index, row in df_experiment.iterrows():
            run_id = row['run_id']
            if experiment_name == config['testing_experiment_name']:
                df_run = pd.read_pickle(
                    'mlruns/{}/{}/artifacts/{}'.format(experiment_id, run_id, 'results.pickle'))
                export_dataframe(
                    df_run, '{}/{}'.format(experiment_name, run_id), config)


def export_dataframe(data, filename, config):
    fullname = filename_to_path('{}/{}'.format(config['filename'], filename))
    if 'csv' in config['format']:
        data.to_csv(str(fullname) + '.csv', index=False)
    if 'pickle' in config['format']:
        data.to_pickle(str(fullname) + '.pickle')
