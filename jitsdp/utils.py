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


def dir_to_path(dirname):
    path = pathlib.Path(dirname)
    path.mkdir(parents=True, exist_ok=True)
    return path


def filename_to_path(filename):
    path = pathlib.Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def setup_and_run(config, frun):
    run_command = standard_run_command(sys.argv)
    meta_model = config['meta_model']
    logging.getLogger('').handlers = []
    dir = pathlib.Path('logs')
    mkdir(dir)
    log = '{}-{}.log'.format(meta_model, datetime.now())
    log = log.replace(' ', '-')
    log = dir / log
    logging.basicConfig(filename=log,
                        filemode='w', level=logging.INFO)
    logging.info('Config: {}'.format(config))

    set_experiment(config)
    with mlflow.start_run():
        mlflow.set_tag('run.command', run_command)
        frun(config=config)
        mlflow.log_artifact(log)


def standard_run_command(sys_args):
    std_args = list(sys_args)
    std_args[0] = pathlib.Path(std_args[0]).name
    return ' '.join(std_args)


def int_or_none(string):
    return None if string == 'None' else int(string)


def unique_dir(config):
    return DIR / '{}_{}_{}_{}_{}_{}'.format(config['rate_driven'], config['meta_model'], config['model'], config['cross_project'], config['dataset'], config['seed'])


def set_experiment(config):
    experiment_name = config['experiment_name']
    if experiment_name is not None:
        mlflow_experiment_id = os.environ.pop('MLFLOW_EXPERIMENT_ID', None)
        if mlflow_experiment_id is not None:
            raise RuntimeError(
                'Use \'mlflow run --experiment-name [name]...\' instead of \'mlflow run ... -Pexperiment-name=[name]\'')
        mlflow.set_experiment(experiment_name)


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


def track_metric(prediction, name, value):
    prediction[name] = value
    return prediction


def track_time(prediction):
    prediction['timestamp_test'] = time.time()
    return prediction
