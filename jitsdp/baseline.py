from jitsdp.data import make_stream, save_results, load_results, DATASETS
from jitsdp.pipeline import set_seed
from jitsdp.utils import mkdir, split_args, create_config_template, to_plural

import argparse
from datetime import datetime
from itertools import product
import logging
import mlflow
import pathlib
import pandas as pd
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Baseline: experiment execution')
    parser.add_argument('--start',   type=int,
                        help='First commit to be used for testing (default: 0).',    default=0)
    parser.add_argument('--cross-project',   type=int,
                        help='Whether must use cross-project data (default: 0).', default=0, choices=[0, 1])
    parser.add_argument('--seeds',   type=int,
                        help='Seeds of random state (default: [0]).',    default=[0], nargs='+')
    parser.add_argument('--datasets',   type=str, help='Datasets to run the experiment. (default: [\'brackets\']).',
                        default=['brackets'], choices=['brackets', 'camel', 'fabric8', 'jgroups', 'neutron', 'tomcat', 'broadleaf', 'nova', 'npm', 'spring-integration'], nargs='+')
    lists = ['seed', 'dataset']
    sys.argv = split_args(sys.argv, lists)
    args = parser.parse_args()
    args = dict(vars(args))
    logging.getLogger('').handlers = []
    dir = pathlib.Path('logs')
    mkdir(dir)
    log = 'baseline-{}.log'.format(datetime.now())
    log = log.replace(' ', '-')
    log = dir / log
    logging.basicConfig(filename=log,
                        filemode='w', level=logging.INFO)
    logging.info('Main config: {}'.format(args))

    mlflow.set_experiment('baseline')
    with mlflow.start_run():
        configs = create_configs(args, lists)
        for config in configs:
            run(config)
        mlflow.log_artifact(log)


def run(config):
    mlflow.log_params(config)
    set_seed(config)
    dataset = config['dataset']
    # stream with commit order
    df_commit = make_stream(dataset)
    # stream with labeling order
    df_test = df_commit.copy()
    df_train = extract_events(df_commit)


def extract_events(df_commit):
    seconds_by_day = 24 * 60 * 60
    # seconds
    verification_latency = 90 * seconds_by_day
    # cleaned
    df_clean = df_commit[df_commit['target'] == 0]
    df_cleaned = df_clean.copy()
    df_cleaned['timestamp_event'] = df_cleaned['timestamp'] + \
        verification_latency
    # bugged
    df_bug = df_commit[df_commit['target'] == 1]
    df_bugged = df_bug.copy()
    df_bugged['timestamp_event'] = df_bugged['timestamp_fix']
    # bug cleaned
    df_bug_cleaned = df_bug.copy()
    waited_time = df_bug_cleaned['timestamp'] - df_bug_cleaned['timestamp_fix']
    df_bug_cleaned = df_bug_cleaned[waited_time >= verification_latency]
    df_bug_cleaned['target'] = 0
    df_bug_cleaned['timestamp_event'] = df_bug_cleaned['timestamp'] + \
        verification_latency
    # events
    df_events = pd.concat([df_cleaned, df_bugged, df_bug_cleaned])
    df_events = df_events.sort_values('timestamp_event')
    print(df_events.head())
    return df_events


def create_configs(args, lists):
    config_template = create_config_template(args, lists)
    plurals = to_plural(lists)
    values_lists = [args[plural] for plural in plurals]
    for values_tuple in product(*values_lists):
        config = dict(config_template)
        for i, name in enumerate(lists):
            config[name] = values_tuple[i]
        yield config
