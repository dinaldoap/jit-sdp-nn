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
    parser.add_argument('--experiment-name',   type=str,
                        help='Experiment name (default: None). None means default behavior of MLflow', default=None)
    parser.add_argument('--pool-size',   type=int,
                        help='Number of processes used to run the experiment in parallel (default: 1).', default=1)
    parser.add_argument('--start',   type=int,
                        help='First commit to be used for testing (default: 0).',    default=0)
    parser.add_argument('--end',   type=int_or_none,
                        help='Last commit to be used for testing (default: 5000). None means all commits.',  default=5000)
    parser.add_argument('--borb-waiting-time',   type=int,
                        help='Number of days to wait before labeling the commit as clean (default: 90).',    default=90)
    parser.add_argument('--borb-ma-window-size',   type=int,
                        help='Number of recent commits to use to calculate the moving average of the model\'s output (default: 100).',    default=100)
    parser.add_argument('--borb-pull-request-size',   type=int,
                        help='Number of commits to wait before retraining (default: 50).',    default=50)
    parser.add_argument('--borb-max-sample-size',   type=int_or_none,
                        help='Max sample size selected from the training data in each iteration (default: 1000). None means bootstrap.',    default=1000)
    parser.add_argument('--borb-th',   type=float,
                        help='Expected value for the moving average of the model\'s output (default: .4).',  default=.4)
    parser.add_argument('--borb-l0',   type=float,
                        help='No description. (default: 10.).',  default=10.)
    parser.add_argument('--borb-l1',   type=float,
                        help='No description. (default: 12.).',  default=12.)
    parser.add_argument('--borb-m',   type=float,
                        help='No description. (default: 1.5).',  default=1.5)
    parser.add_argument('--borb',   type=int,
                        help='Whether must use oversampling rate boosting to balance output proportions (default: 1).', default=1, choices=[0, 1])
    parser.add_argument('--cross-project',   type=int,
                        help='Whether must use cross-project data (default: 0).', default=0, choices=[0, 1])
    parser.add_argument('--seeds',   type=int,
                        help='Seeds of random state (default: [0]).',    default=[0], nargs='+')
    parser.add_argument('--datasets',   type=str, help='Datasets to run the experiment. (default: [\'brackets\']).',
                        default=['brackets'], choices=['brackets', 'camel', 'fabric8', 'jgroups', 'neutron', 'tomcat', 'broadleaf', 'nova', 'npm', 'spring-integration'], nargs='+')
    parser.add_argument('--models',   type=str,
                        help='Which models must use in the ensemble (default: [\'mlp\']).', default=['mlp'], choices=['bht', 'obht', 'lr', 'mlp', 'nb', 'rf', 'svm'], nargs='+')
    parser.add_argument('--bht-n-estimators',   type=int,
                        help='The number of hoeffding trees (default: 1).',    default=1)
    parser.add_argument('--lr-alpha',   type=float,
                        help='Constant that multiplies the regularization term. Also used to compute the learning rate (default: .1).',  default=.1)
    parser.add_argument('--lr-l1-ratio',   type=float,
                        help='The Elastic Net mixing parameter (default: .15).',  default=.15)
    parser.add_argument('--lr-n-epochs',   type=int,
                        help='Number of epochs performed by the training (default: 1).',    default=1)
    parser.add_argument('--lr-batch-size',   type=int,
                        help='Number of commits included in each batch (default: 256).',    default=256)
    parser.add_argument('--mlp-n-hidden-layers',   type=int,
                        help='Number of hidden layers (default: 1).',    default=1)
    parser.add_argument('--mlp-hidden-layers-size',   type=int,
                        help='Hidden layers size (default: 7).',    default=7)
    parser.add_argument('--mlp-learning-rate',   type=float,
                        help='Learning rate (default: .001).',  default=.001)
    parser.add_argument('--mlp-n-epochs',   type=int,
                        help='Number of epochs performed by the training (default: 1).',    default=1)
    parser.add_argument('--mlp-dropout-input-layer',   type=float,
                        help='Dropout probability of the input layer (default: .2).',    default=.2)
    parser.add_argument('--mlp-dropout-hidden-layers',   type=float,
                        help='Dropout probability of the hidden layers (default: .5).',    default=.5)
    parser.add_argument('--mlp-batch-size',   type=int,
                        help='Number of commits included in each batch (default: 256).',    default=256)
    parser.add_argument('--nb-n-updates',   type=int,
                        help='Number of updates performed by the training (default: 1).',    default=1)
    parser.add_argument('--rf-n-estimators',   type=int,
                        help='The number of trees in the forest (default: 1).',    default=1)
    parser.add_argument('--rf-criterion',   type=str,
                        help='The function to measure the quality of a split (default: \'entropy\').', default='entropy', choices=['entropy', 'gini'])
    parser.add_argument('--rf-max-depth',   type=int,
                        help='The maximum depth of the tree (default: unlimited).', default=None)
    parser.add_argument('--rf-max-features',   type=int,
                        help='The number of features to consider when looking for the best split (default: 3).', default=3)
    parser.add_argument('--rf-min-samples-leaf',   type=int,
                        help='The minimum number of samples required to be at a leaf node (default: 200).', default=200)
    parser.add_argument('--rf-min-impurity-decrease',   type=float,
                        help='A node will be split if this split induces a decrease of the impurity greater than or equal to this value (default: .0).', default=.0)
    parser.add_argument('--svm-alpha',   type=float,
                        help='Constant that multiplies the regularization term. Also used to compute the learning rate (default: .1).',  default=.1)
    parser.add_argument('--svm-l1-ratio',   type=float,
                        help='The Elastic Net mixing parameter (default: .15).',  default=.15)
    parser.add_argument('--svm-n-epochs',   type=int,
                        help='Number of epochs performed by the training (default: 1).',    default=1)
    parser.add_argument('--svm-batch-size',   type=int,
                        help='Number of commits included in each batch (default: 256).',    default=256)
    parser.add_argument('--track-time',   type=int,
                        help='Whether must track time. (default: 0).',  default=0)
    parser.add_argument('--track-rf',   type=int,
                        help='Whether must track random forest complexity. (default: 0).',  default=0)
    parser.add_argument('--f-val',   type=float,
                        help='Fraction of labeled data to be used for validation. (default: .0).',  default=.0)
    parser.add_argument('--ensemble-size',   type=int,
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

    if args['experiment_name'] is not None:
        mlflow.set_experiment(args['experiment_name'])
    del args['experiment_name']

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
