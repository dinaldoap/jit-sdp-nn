from jitsdp.evaluation import run, report

import logging
import argparse
import mlflow


def args_to_config(args):
    config = dict(vars(args))
    del config['datasets']
    return config


def main():
    parser = argparse.ArgumentParser(
        description='JIT-SDP: experiment execution')
    parser.add_argument('command',   type=str, help='Which command should execute (default: run).',
                        default='run', choices=['run', 'report'])
    parser.add_argument('--epochs',   type=int,
                        help='Number of epochs performed by the training (default: 1).',    default=1)
    parser.add_argument('--start',   type=int,
                        help='First commit to be used for testing (default: 1000). A minimum of one fold is not used despite this parameter.',    default=1000)
    parser.add_argument('--folds',   type=float,
                        help='Fraction of folds to be used by the evaluation. A minimum of two folds is always used despite this parameter. (default: 0).',  default=0)
    parser.add_argument('--fold_size',   type=int,
                        help='Number of commits in each fold (default: 50).',    default=50)
    parser.add_argument('--datasets',   type=str, help='Datasets to run the experiment. (default: brackets).',
    sys.argv = split_arg(sys.argv, '--datasets')
    args = parser.parse_args()
    print('Configuration: {}'.format(args))
    logging.getLogger('').handlers = []
    logging.basicConfig(filename='logs/jitsdp.log',
                        filemode='w', level=logging.DEBUG)
    commands = {
        'run': run,
        'report': report,
    }
    command = commands[args.command]
    args_config = args_to_config(args)
    for dataset in args.datasets:
        config = dict(args_config)
        config['dataset'] = dataset
        with mlflow.start_run():
            command(config=config)


if __name__ == '__main__':
    main()
