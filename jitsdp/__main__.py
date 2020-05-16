from jitsdp.evaluation import run, report
from jitsdp.utils import mkdir, split_args, create_config_template, to_plural

import argparse
from datetime import datetime
from itertools import product
import logging
import mlflow
from multiprocessing import Pool
import pathlib
import sys


def main():
    parser = argparse.ArgumentParser(
        description='JIT-SDP: experiment execution')
    parser.add_argument('command',   type=str, help='Which command should execute (default: run).',
                        default='run', choices=['run', 'report'])
    parser.add_argument('--pool_size',   type=int,
                        help='Number of processes used to run the experiment in parallel (default: 1).', default=1)
    parser.add_argument('--seeds',   type=int,
                        help='Seeds of random state (default: [0]).',    default=[0], nargs='+')
    # TODO: rename to n_epochs
    parser.add_argument('--epochs',   type=int,
                        help='Number of epochs performed by the training (default: 1).',    default=1)
    # TODO: add --n_trees
    parser.add_argument('--start',   type=int,
                        help='First commit to be used for testing (default: 0). The first fold is not used despite this parameter.',    default=0)
    # TODO: rename to f_folds and replace 0 to 0.0 0.0
    parser.add_argument('--folds',   type=float,
                        help='Fraction of folds to be used by the evaluation. A minimum of two folds is always used despite this parameter. (default: 0).',  default=0)
    parser.add_argument('--fold_size',   type=int,
                        help='Number of commits in each fold (default: 50).',    default=50)
    parser.add_argument('--normal_proportion',   type=float,
                        help='Expected proportion for normal commits. (default: .6).',  default=.6)
    # TODO: make zero disables ensemble and default to zero
    parser.add_argument('--ensemble_size',   type=int,
                        help='Number of models in the ensemble (default: 1).',    default=1)
    parser.add_argument('--models',   type=str,
                        help='Which models must use in the ensemble (default: [\'mlp\']).', default=['mlp'], choices=['mlp', 'nb', 'rf', 'sgd'], nargs='+')
    parser.add_argument('--orb',   type=int,
                        help='Whether must use oversampling rate boosting to balance output proportions (default: 0).', default=0, choices=[0, 1])
    parser.add_argument('--threshold',   type=int,
                        help='Whether must tune threshold to balance output proportions (default: 0).', default=0, choices=[0, 1, 2])
    parser.add_argument('--uncertainty',   type=int,
                        help='Whether must use decreasing uncertainty about normal commit labels inside verification latency (default: 0).', default=0, choices=[0, 1])
    parser.add_argument('--incremental',   type=int,
                        help='Whether must do incremental training along the stream (default: 0).', default=0, choices=[0, 1])
    parser.add_argument('--datasets',   type=str, help='Datasets to run the experiment. (default: [\'brackets\']).',
                        default=['brackets'], choices=['brackets', 'camel', 'fabric8', 'jgroups', 'neutron', 'tomcat', 'broadleaf', 'nova', 'npm', 'spring-integration'], nargs='+')
    lists = ['seed', 'dataset', 'model']
    sys.argv = split_args(sys.argv, lists)
    args = parser.parse_args()
    args = dict(vars(args))
    logging.getLogger('').handlers = []
    dir = pathlib.Path('logs')
    mkdir(dir)
    log = 'jitsdp-{}.log'.format(datetime.now())
    log = log.replace(' ', '-')
    log = dir / log
    logging.basicConfig(filename=log,
                        filemode='w', level=logging.INFO)
    logging.info('Main config: {}'.format(args))
    with mlflow.start_run():
        configs = create_configs(args, lists)
        with Pool(args['pool_size']) as pool:
            pool.map(safe_run, configs)
        mlflow.log_artifact(log)


def safe_run(config):
    try:
        run_nested(config)
    except Exception:
        logging.exception('Exception raised on config: {}'.format(config))


def run_nested(config):
    commands = {
        'run': run,
        'report': report,
    }
    command = commands[config['command']]
    with mlflow.start_run(nested=True):
        logging.info('Nested config: {}'.format(config))
        command(config=config)


def create_configs(args, lists):
    config_template = create_config_template(args, lists)
    plurals = to_plural(lists)
    values_lists = [args[plural] for plural in plurals]
    for values_tuple in product(*values_lists):
        config = dict(config_template)
        for i, name in enumerate(lists):
            config[name] = values_tuple[i]
        yield config


if __name__ == '__main__':
    main()
