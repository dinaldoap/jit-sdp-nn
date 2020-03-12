from jitsdp.evaluation import run, report
from jitsdp.utils import split_arg, mkdir

import argparse
import logging
import mlflow
import pathlib
import sys


def args_to_config(args):
    config = dict(vars(args))
    del config['datasets']
    return config


def main():
    parser = argparse.ArgumentParser(
        description='JIT-SDP: experiment execution')
    parser.add_argument('command',   type=str, help='Which command should execute (default: run).',
                        default='run', choices=['run', 'report'])
    parser.add_argument('--seed',   type=int,
                        help='Seed of random state (default: 42).',    default=42)
    parser.add_argument('--epochs',   type=int,
                        help='Number of epochs performed by the training (default: 1).',    default=1)
    parser.add_argument('--start',   type=int,
                        help='First commit to be used for testing (default: 0). The first fold is not used despite this parameter.',    default=0)
    parser.add_argument('--folds',   type=float,
                        help='Fraction of folds to be used by the evaluation. A minimum of two folds is always used despite this parameter. (default: 0).',  default=0)
    parser.add_argument('--fold_size',   type=int,
                        help='Number of commits in each fold (default: 50).',    default=50)
    parser.add_argument('--normal_proportion',   type=float,
                        help='Expected proportion for normal commits. (default: .6).',  default=.6)
    parser.add_argument('--estimators',   type=int,
                        help='Number of estimators in the ensemble (default: 1).',    default=1)
    parser.add_argument('--orb',   type=int,
                        help='Whether must use oversampling rate boosting to balance output proportions (default: 0).', default=0, choices=[0, 1])
    parser.add_argument('--threshold',   type=int,
                        help='Whether must tune threshold to balance output proportions (default: 0).', default=0, choices=[0, 1, 2])
    parser.add_argument('--uncertainty',   type=int,
                        help='Whether must use decreasing uncertainty about normal commit labels inside verification latency (default: 0).', default=0, choices=[0, 1])
    parser.add_argument('--incremental',   type=int,
                        help='Whether must do incremental training along the stream (default: 0).', default=0, choices=[0, 1])
    parser.add_argument('--datasets',   type=str, help='Datasets to run the experiment. (default: brackets).',
                        default=['brackets'], choices=['brackets', 'camel', 'fabric8', 'jgroups', 'neutron', 'tomcat'], nargs='+')
    sys.argv = split_arg(sys.argv, '--datasets')
    args = parser.parse_args()
    print('Configuration: {}'.format(args))
    logging.getLogger('').handlers = []
    dir = pathlib.Path('logs')
    mkdir(dir)
    logging.basicConfig(filename=dir / 'jitsdp.log',
                        filemode='w', level=logging.DEBUG)
    commands = {
        'run': run,
        'report': report,
    }
    command = commands[args.command]
    args_config = args_to_config(args)
    with mlflow.start_run():
        for dataset in args.datasets:
            config = dict(args_config)
            config['dataset'] = dataset
            with mlflow.start_run(nested=True):
                command(config=config)


if __name__ == '__main__':
    main()
