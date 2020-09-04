# coding=utf-8
from jitsdp.constants import DIR

from datetime import datetime
from itertools import product
import logging
import mlflow
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import pathlib
import sys
import time


def mkdir(dir):
    dir.mkdir(parents=True, exist_ok=True)


def setup_and_run(parser, logname, frun):
    run_command = ' '.join(sys.argv)    
    lists = ['seed', 'dataset', 'model']
    sys.argv = split_args(sys.argv, lists)
    args = parser.parse_args()
    args = dict(vars(args))
    logging.getLogger('').handlers = []
    dir = pathlib.Path('logs')
    mkdir(dir)
    log = '{}-{}.log'.format(logname, datetime.now())
    log = log.replace(' ', '-')
    log = dir / log
    logging.basicConfig(filename=log,
                        filemode='w', level=logging.INFO)
    logging.info('Main config: {}'.format(args))

    set_experiment(args)
    args['frun'] = frun
    with mlflow.start_run():
        mlflow.set_tag('run.command', run_command)
        configs = create_configs(args, lists)
        with Pool(args['pool_size']) as pool:
            codes = pool.map(safe_run, configs)
        mlflow.log_artifact(log)
        return sum(codes)


def safe_run(config):
    try:
        run_nested(config)
        return 0
    except Exception:
        logging.exception('Exception raised on config: {}'.format(config))
        return 1


def run_nested(config):
    frun = config['frun']
    del config['frun']
    with mlflow.start_run(nested=True):
        logging.info('Nested config: {}'.format(config))
        frun(config=config)


def create_configs(args, lists):
    config_template = create_config_template(args, lists)
    plurals = to_plural(lists)
    values_lists = [args[plural] for plural in plurals]
    for values_tuple in product(*values_lists):
        config = dict(config_template)
        for i, name in enumerate(lists):
            config[name] = values_tuple[i]
        yield config


def int_or_none(string):
    return None if string == 'None' else int(string)


def split_args(argv, names):
    cli_tokens = to_cli_tokens(names)
    new_argv = argv
    for cli_token in cli_tokens:
        new_argv = split_arg(new_argv, cli_token)
    return new_argv


def to_cli_tokens(names):
    return ['--{}s'.format(name) for name in names]


def split_arg(argv, name):
    try:
        value_index = argv.index(name) + 1
    except ValueError:  # argument not in list
        return argv
    value = argv[value_index]
    return argv[:value_index] + value.split() + argv[value_index + 1:]


def create_config_template(args, names):
    new_args = dict(args)
    plurals = to_plural(names)
    for plural_name in plurals:
        del new_args[plural_name]
    del new_args['experiment_name']
    return new_args


def to_plural(names):
    return ['{}s'.format(name) for name in names]


def create_configs(args, lists):
    config_template = create_config_template(args, lists)
    plurals = to_plural(lists)
    values_lists = [args[plural] for plural in plurals]
    for values_tuple in product(*values_lists):
        config = dict(config_template)
        for i, name in enumerate(lists):
            config[name] = values_tuple[i]
        yield config


def unique_dir(config):
    return DIR / '{}_{}_{}'.format(config['seed'], config['dataset'], config['model'])


def set_experiment(args):
    experiment_name = args['experiment_name']
    if experiment_name is not None:
        mlflow_experiment_id = os.environ.pop('MLFLOW_EXPERIMENT_ID', None)
        if mlflow_experiment_id is not None:
            raise RuntimeError(
                'Use \'mlflow run --experiment-name [name]...\' instead of \'mlflow run ... -Pexperiment-name=[name]\'')
        os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name


def track_forest(prediction, forest):
    properties = {
        'depth': lambda tree: tree.get_depth() if forest.trained else 0,
        'n_leaves': lambda tree: tree.get_n_leaves() if forest.trained else 0,
    }
    for name, func in properties.items():
        values = _extract_property(forest, func)
        prediction = _concat_property(prediction, name, values)
    return prediction


def _extract_property(forest, func):
    if forest.trained:
        return [func(estimator) for estimator in forest.estimators]
    else:
        return [0.]


def _concat_property(prediction, name, values):
    prop = pd.Series(values, dtype=np.float64)
    prop = prop.describe()
    prop = prop.to_frame()
    prop = prop.transpose()
    prop.columns = ['{}_{}'.format(name, column) for column in prop.columns]
    template = [prop.head(0)]
    prop = pd.concat(template + [prop] * len(prediction))
    prop.index = prediction.index
    return pd.concat([prediction, prop], axis='columns')


def track_time(prediction):
    prediction['timestamp_test'] = time.time()
    return prediction
