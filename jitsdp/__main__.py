# coding=utf-8
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
    parser.add_argument('--start',   type=int,
                        help='First commit to be used for testing (default: 0).',    default=0)
    parser.add_argument('--end',   type=int_or_none,
                        help='Last commit to be used for testing (default: 5000). None means all commits.',  default=5000)
    parser.add_argument('--pull_request_size',   type=int,
                        help='Number of commits to wait before retraining (default: 50).',    default=50)
    parser.add_argument('--max_sample_size',   type=int_or_none,
                        help='Max sample size selected from the training data in each iteration (default: 1000). None means bootstrap.',    default=1000)
    parser.add_argument('--normal_proportion',   type=float,
                        help='Expected proportion for normal commits. (default: .6).',  default=.6)
    parser.add_argument('--orb',   type=int,
                        help='Whether must use oversampling rate boosting to balance output proportions (default: 0).', default=0, choices=[0, 1])
    parser.add_argument('--cross_project',   type=int,
                        help='Whether must use cross-project data (default: 0).', default=0, choices=[0, 1])
    parser.add_argument('--seeds',   type=int,
                        help='Seeds of random state (default: [0]).',    default=[0], nargs='+')
    parser.add_argument('--datasets',   type=str, help='Datasets to run the experiment. (default: [\'brackets\']).',
                        default=['brackets'], choices=['brackets', 'camel', 'fabric8', 'jgroups', 'neutron', 'tomcat', 'broadleaf', 'nova', 'npm', 'spring-integration'], nargs='+')
    parser.add_argument('--models',   type=str,
                        help='Which models must use in the ensemble (default: [\'mlp\']).', default=['mlp'], choices=['lr', 'mlp', 'nb', 'rf', 'svm'], nargs='+')
    parser.add_argument('--lr_alpha',   type=float,
                        help='Constant that multiplies the regularization term. Also used to compute the learning rate (default: .1).',  default=.1)
    parser.add_argument('--lr_l1_ratio',   type=float,
                        help='The Elastic Net mixing parameter (default: .15).',  default=.15)
    parser.add_argument('--lr_n_epochs',   type=int,
                        help='Number of epochs performed by the training (default: 1).',    default=1)
    parser.add_argument('--lr_batch_size',   type=int,
                        help='Number of commits included in each batch (default: 512).',    default=512)
    parser.add_argument('--mlp_n_hidden_layers',   type=int,
                        help='Number of hidden layers (default: 1).',    default=1)
    parser.add_argument('--mlp_learning_rate',   type=float,
                        help='Learning rate (default: .001).',  default=.001)
    parser.add_argument('--mlp_n_epochs',   type=int,
                        help='Number of epochs performed by the training (default: 1).',    default=1)
    parser.add_argument('--nb_n_updates',   type=int,
                        help='Number of updates performed by the training (default: 1).',    default=1)
    parser.add_argument('--rf_n_estimators',   type=int,
                        help='The number of trees in the forest (default: 1).',    default=1)
    parser.add_argument('--rf_criterion',   type=str,
                        help='The function to measure the quality of a split (default: \'entropy\').', default='entropy', choices=['entropy', 'gini'])
    parser.add_argument('--rf_max_depth',   type=int,
                        help='The maximum depth of the tree (default: 3).', default=3)
    parser.add_argument('--rf_max_features',   type=int,
                        help='The number of features to consider when looking for the best split (default: 3).', default=3)
    parser.add_argument('--rf_min_samples_leaf',   type=int,
                        help='he minimum number of samples required to be at a leaf node (default: 100).', default=100)
    parser.add_argument('--rf_min_impurity_decrease',   type=float,
                        help='A node will be split if this split induces a decrease of the impurity greater than or equal to this value (default: .02).', default=.02)
    parser.add_argument('--svm_alpha',   type=float,
                        help='Constant that multiplies the regularization term. Also used to compute the learning rate (default: .1).',  default=.1)
    parser.add_argument('--svm_l1_ratio',   type=float,
                        help='The Elastic Net mixing parameter (default: .15).',  default=.15)
    parser.add_argument('--svm_n_epochs',   type=int,
                        help='Number of epochs performed by the training (default: 1).',    default=1)
    parser.add_argument('--f_val',   type=float,
                        help='Fraction of labeled data to be used for validation. (default: .0).',  default=.0)
    parser.add_argument('--ensemble_size',   type=int,
                        help='Number of models in the ensemble (default: 1).',    default=1)
    parser.add_argument('--threshold',   type=int,
                        help='Whether must tune threshold to balance output proportions (default: 0).', default=0, choices=[0, 1, 2])
    parser.add_argument('--uncertainty',   type=int,
                        help='Whether must use decreasing uncertainty about normal commit labels inside verification latency (default: 0).', default=0, choices=[0, 1])
    parser.add_argument('--incremental',   type=int,
                        help='Whether must do incremental training along the stream (default: 0).', default=0, choices=[0, 1])
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


def int_or_none(string):
    return None if string == 'None' else int(string)


if __name__ == '__main__':
    sys.exit(main())
