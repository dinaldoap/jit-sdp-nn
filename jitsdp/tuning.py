# coding=utf-8
from jitsdp.utils import filename_to_path, random_state_seed

import argparse
import itertools
import numpy as np
from hyperopt import hp
from hyperopt.pyll.base import scope
import hyperopt.pyll.stochastic as config_space_sampler


class Experiment():

    def __init__(self, validation_size, bundle, experiment_config, seed_dataset_configs, models_configs):
        self.validation_size = validation_size
        self.bundle = bundle
        self.meta_model = experiment_config['meta-model']
        self.experiment_config = experiment_config
        self.seed_dataset_configs = seed_dataset_configs
        self.models_configs = models_configs

    def remove_meta_model(self, config):
        config = dict(config)
        del config['meta-model']
        return config

    def to_configs(self):
        configs = []
        for models_config in self.models_configs:
            for seed_dataset_config in self.seed_dataset_configs:
                config = dict()
                config.update(self.experiment_config)
                config.update(models_config)
                config.update(seed_dataset_config)
                config = self.add_start(config)
                configs.append(config)
        return configs

    def to_shell(self, out):
        for config in self.to_configs():
            config = self.remove_meta_model(config)
            params = ['--{} {}'.format(key, value)
                      for key, value in config.items()]
            params = ' '.join(params)
            prefix = './' if self.bundle else ''
            out.write(
                '{}jitsdp {} {}\n'.format(prefix, self.meta_model, params))

    def add_start(self, config):
        config = dict(config)
        config['end'] = self.validation_size[config['cross-project']]
        return config


def add_arguments(parser, filename):
    add_shared_arguments(parser, filename)
    parser.add_argument('--bundle', type=int,
                        help='Whether must generate commands to the bundle executable.', default=0, choices=[0, 1])
    parser.add_argument('--validation-size', type=int,
                        help='Number of commits used for hyperparameter tuning. This list will be ziped with the cross-project list.', required=True, nargs='+')


def add_shared_arguments(parser, filename):
    parser.add_argument('--start',   type=int,
                        help='Starting index of the random configurations slice.', required=True)
    parser.add_argument('--end',   type=int,
                        help='Stopping index of the random configurations slice.', required=True)
    parser.add_argument('--cross-project',   type=int,
                        help='Whether must use cross-project data.', required=True, choices=[0, 1], nargs='+')
    parser.add_argument('--filename',   type=str,
                        help='Output script path.', default=filename)


def generate(config):
    # experiments
    cross_project = config['cross_project']
    validation_size = config['validation_size']
    assert len(cross_project) == len(
        validation_size), 'cross-project list must match validation-size list in terms of length.'
    validation_size = dict(zip(cross_project, validation_size))
    orb_grid = {
        'meta-model': ['orb'],
        'cross-project': cross_project,
        'model': ['oht'],
    }
    borb_grid = {
        'meta-model': ['borb'],
        'cross-project': cross_project,
        'model': ['ihf', 'lr', 'mlp', 'nb', 'irf'],
    }
    experiment_configs = [
        borb_grid,
        orb_grid,
    ]
    # seeds and datasets
    experiment_configs = map(grid_to_configs, experiment_configs)
    experiment_configs = itertools.chain.from_iterable(experiment_configs)
    seed_dataset_configs = {
        'dataset': ['brackets', 'camel', 'fabric8', 'jgroups', 'neutron', 'tomcat', 'broadleaf', 'nova', 'npm', 'spring-integration'],
        'seed': [118819124794768324716243582738038647832, 233788382964979925575822780126624241621, 123852561530946589675929508680442328351],
    }
    seed_dataset_configs = grid_to_configs(seed_dataset_configs)
    # meta-models and models
    models_configs = create_models_configs(config)
    file_ = filename_to_path(config['filename'])
    with open(file_, mode='w') as out:
        for experiment in configs_to_experiments(validation_size, config['bundle'], experiment_configs, seed_dataset_configs, models_configs):
            experiment.to_shell(out)


def configs_to_experiments(validation_size, bundle, experiment_configs, seed_dataset_configs, models_configs):
    for experiment_config in experiment_configs:
        model = experiment_config['model']
        experiment = Experiment(validation_size=validation_size,
                                bundle=bundle,
                                experiment_config=experiment_config,
                                seed_dataset_configs=seed_dataset_configs, models_configs=models_configs[model])
        yield experiment


def grid_to_configs(grid):
    keys = grid.keys()
    values_lists = grid.values()
    values_tuples = itertools.product(*values_lists)
    return list(map(lambda values_tuple: dict(zip(keys, list(values_tuple))), values_tuples))


def create_models_configs(config):
    meta_model_shared = meta_model_shared_config_space()
    orb = {}
    orb.update(meta_model_shared['orb'])
    orb.update([loguniform('orb-decay-factor', .9, .999),
                uniform('orb-n', 3, 7),
                uniform('orb-rd-grace-period', 100, 500),
                ])

    hoeffding_shared = hoeffding_shared_config_space(config)
    oht = {}
    oht.update(orb)
    oht.update(hoeffding_shared['oht'])

    borb = {}
    borb.update(meta_model_shared['borb'])
    borb.update([uniform('borb-pull-request-size', 100, 500),
                 uniform('borb-sample-size', 1000,
                         8000 if 1 in config['cross_project'] else 4000),
                 ])

    ihf = {}
    ihf.update(borb)
    ihf.update(hoeffding_shared['ihf'])

    linear_shared = linear_shared_config_space()
    lr = {}
    lr.update(borb)
    lr.update(linear_shared['lr'])
    lr.update([
        loguniform('lr-alpha', .01, 1.),
    ])

    mlp = {}
    mlp.update(borb)
    mlp.update(linear_shared['mlp'])
    mlp.update([
        loguniform('mlp-learning-rate', .0001, .01),
        uniform('mlp-n-hidden-layers', 1, 3),
        uniform('mlp-hidden-layers-size', 5, 15),
        uniform('mlp-dropout-input-layer', .1, .3),
        uniform('mlp-dropout-hidden-layer', .3, .5),
    ])

    nb = {}
    nb.update(borb)
    nb.update([
        uniform('nb-n-updates', 10, 80),
    ])

    irf = {}
    irf.update(borb)
    irf.update([
        uniform('irf-n-estimators', 20, 100),
        choiceuniform('irf-criterion', ['gini', 'entropy']),
        uniform('irf-min-samples-leaf', 100, 300),
        uniform('irf-max-features', 3, 7),
    ])

    start = config['start']
    end = config['end']
    models_configs = {'oht': config_space_to_configs(oht, start, end),
                      'ihf': config_space_to_configs(ihf, start, end),
                      'lr': config_space_to_configs(lr, start, end),
                      'mlp': config_space_to_configs(mlp, start, end),
                      'nb': config_space_to_configs(nb, start, end),
                      'irf': config_space_to_configs(irf, start, end),
                      }

    return models_configs


def uniform(name, start, end):
    if is_int(start):
        return (name, to_int(start + hp.quniform(name, 0, end - start, 1)))
    else:
        return (name, hp.uniform(name, start, end))


def loguniform(name, start, end):
    if is_int(start):
        return (name, to_int(start + hp.qloguniform(name, 0, np.log(end) - np.log(start), 1)))
    else:
        return (name, hp.loguniform(name, np.log(start), np.log(end)))


def is_int(start):
    return int == type(start)


def to_int(apply):
    return scope.int(apply)


def choiceuniform(name, options):
    return (name, hp.choice(name, options))


def meta_model_shared_config_space():
    config_spaces = {}
    meta_models = ['orb', 'borb']
    for meta_model in meta_models:
        config_spaces[meta_model] = [
            uniform('{}-waiting-time'.format(meta_model), 90, 180),
            uniform('{}-ma-window-size'.format(meta_model), 50, 200),
            uniform('{}-th'.format(meta_model), .3, .5),
            loguniform('{}-l0'.format(meta_model), 1., 20.),
            loguniform('{}-l1'.format(meta_model), 1., 20.),
            uniform('{}-m'.format(meta_model), 1.1, np.e),
        ]
    return config_spaces


def hoeffding_shared_config_space(config):
    config_spaces = {}
    models = ['oht', 'ihf']
    max_n_estimators = {
        'oht': 40,
        'ihf': 30,
    }
    for model in models:
        config_spaces[model] = [
            uniform('{}-n-estimators'.format(model),
                    10, max_n_estimators[model]),
            uniform('{}-grace-period'.format(model), 100, 500),
            choiceuniform('{}-split-criterion'.format(model),
                          ['gini', 'info_gain', 'hellinger']),
            loguniform('{}-split-confidence'.format(model), 0.0000001, 0.5),
            uniform('{}-tie-threshold'.format(model), 0.05, 0.5),
            # use only False (default) to avoid bug when value is True
            #choiceuniform('{}-remove-poor-atts'.format(model), [1, 0]),
            choiceuniform('{}-no-preprune'.format(model), [1, 0]),
            choiceuniform('{}-leaf-prediction'.format(model),
                          ['mc', 'nb', 'nba']),
        ]
    return config_spaces


def linear_shared_config_space():
    config_spaces = {}
    models = ['lr', 'mlp']
    for model in models:
        config_spaces[model] = [
            uniform('{}-n-epochs'.format(model),  10, 80),
            loguniform('{}-batch-size'.format(model), 128, 512),
            choiceuniform('{}-log-transformation'.format(model), [0, 1]),
        ]
    return config_spaces


def config_space_to_configs(config_space, start, end):
    rng = np.random.RandomState(seed=random_state_seed(
        168584965791772849512190648581246426632))
    configs = [config_space_sampler.sample(
        config_space, rng=rng) for i in range(end)]
    return configs[start:end]
