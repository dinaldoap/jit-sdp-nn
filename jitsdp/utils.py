# coding=utf-8
from jitsdp.constants import DIR

from itertools import product
import mlflow
import os


def mkdir(dir):
    dir.mkdir(parents=True, exist_ok=True)


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
    mlflow_exp_id = os.environ.pop('MLFLOW_EXPERIMENT_ID', None)
    project_exp_id = args['experiment_name']
    if mlflow_exp_id is None and project_exp_id is not None:
        mlflow.set_experiment(project_exp_id)
    del args['experiment_name']
