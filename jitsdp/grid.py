# coding=utf-8
import itertools
import numpy as np
from hyperopt import hp
import hyperopt.pyll.stochastic as config_space_sampler


class Experiment():

    def __init__(self, experiment_config, seed_dataset_configs, models_configs):
        self.experiment_config = self.fix_experimnt_config(experiment_config)
        self.seed_dataset_configs = seed_dataset_configs
        self.models_configs = models_configs

    def fix_experimnt_config(self, experiment_config):
        self.rate_driven = experiment_config['rate-driven']
        self.meta_model = experiment_config['meta-model']
        new_experiment_config = dict(experiment_config)
        del new_experiment_config['rate-driven']
        del new_experiment_config['meta-model']
        return new_experiment_config

    @property
    def name(self):
        rate_driven = 'r' if self.rate_driven else ''
        model = self.experiment_config['models']
        train_data = 'cp' if self.experiment_config['cross-project'] else 'wp'
        return '{}{}-{}-{}'.format(rate_driven, self.meta_model, model, train_data)

    def to_shell(self, out):
        for seed_dataset_config in self.seed_dataset_configs:
            for models_config in self.models_configs:
                config = dict()
                config.update(self.experiment_config)
                config.update(seed_dataset_config)
                config.update(models_config)
                config = self.fix_rate_driven(config)
                config = self.add_start(config)
                params = ['--{} {}'.format(key, value)
                          for key, value in config.items()]
                params = ' '.join(params)
                out.write(
                    './jitsdp {} {}\n'.format(self.meta_model, params))

    def fix_rate_driven(self, config):
        config = dict(config)
        config['{}-rd'.format(self.meta_model)] = self.rate_driven
        return config

    def add_start(self, config):
        config = dict(config)
        config['end'] = 1000 if config['cross-project'] else 5000
        return config


def main():
    # experiments
    orb_rorb_grid = {
        'meta-model': ['orb'],
        'cross-project': [0, 1],
        'rate-driven': [0, 1],
        'models': ['hts'],
    }
    borb_rborb_grid = {
        'meta-model': ['borb'],
        'cross-project': [0, 1],
        'rate-driven': [0, 1],
        'models': ['ihf'],
    }
    rborb_grid = {
        'meta-model': ['borb'],
        'cross-project': [0, 1],
        'rate-driven': [1],
        'models': ['lr', 'mlp', 'nb', 'irf'],
    }
    experiment_configs = [
        orb_rorb_grid,
        borb_rborb_grid,
        rborb_grid,
    ]
    # seeds and datasets
    experiment_configs = map(grid_to_configs, experiment_configs)
    experiment_configs = itertools.chain.from_iterable(experiment_configs)
    seed_dataset_configs = {
        'seeds': [0, 1, 2, 3, 4],
        'datasets': ['brackets', 'camel', 'fabric8', 'jgroups', 'neutron', 'tomcat', 'broadleaf', 'nova', 'npm', 'spring-integration'],
    }
    seed_dataset_configs = grid_to_configs(seed_dataset_configs)
    # meta-models and models
    models_configs = create_models_configs()

    with open('jitsdp/dist/tuning.sh', mode='w') as out:
        for experiment in configs_to_experiments(experiment_configs, seed_dataset_configs, models_configs):
            experiment.to_shell(out)


def configs_to_experiments(experiment_configs, seed_dataset_configs, models_configs):
    for experiment_config in experiment_configs:
        model = experiment_config['models']
        experiment = Experiment(experiment_config=experiment_config,
                                seed_dataset_configs=seed_dataset_configs, models_configs=models_configs[model])
        yield experiment


def grid_to_configs(grid):
    keys = grid.keys()
    values_lists = grid.values()
    values_tuples = itertools.product(*values_lists)
    return list(map(lambda values_tuple: dict(zip(keys, list(values_tuple))), values_tuples))


def create_models_configs():
    shared_config_spaces = meta_model_shared_config_space()
    orb = {}
    orb.update(shared_config_spaces['orb'])
    orb.update([loguniform('orb-decay-factor', .9, .999),
                uniform('orb-n', 3, 7, 2),
                uniform('orb-rd-grace-period', 100, 500, 100),                
                ])

    hts = {}
    hts.update(orb)
    hts.update([uniform('hts-n-estimators', 20, 100, 20),
                uniform('hts-grace-period', 100, 500, 100),
                choiceuniform('hts-split-criterion',
                              ['gini', 'info_gain', 'hellinger']),
                loguniform('hts-split-confidence', 0.0000001, 0.5),
                uniform('hts-tie-threshold', 0.05, 0.5),
                choiceuniform('hts-remove-poor-atts', [1, 0]),
                choiceuniform('hts-no-preprune', [1, 0]),
                choiceuniform('hts-leaf-prediction', ['mc', 'nb', 'nba']),
                ])

    models_configs = {'hts': config_space_to_configs(hts, start=0, end=10),
                      'ihf': [],
                      'lr': [],
                      'mlp': [],
                      'nb': [],
                      'irf': [],
                      }

    return models_configs


def uniform(name, start, end, step=None):
    if step is None:
        return (name, hp.uniform(name, start, end))
    else:
        return (name, start + hp.quniform(name, 0, end - start, step))


def loguniform(name, start, end, step=None):
    if step is None:
        return (name, hp.loguniform(name, np.log(start), np.log(end)))
    else:
        return (name, start + hp.qloguniform(name, 0, np.log(end) - np.log(start), step))


def choiceuniform(name, options):
    return (name, hp.choice(name, options))

def meta_model_shared_config_space():    
    config_spaces = {}
    meta_models = ['orb', 'borb']
    for meta_model in meta_models:
        config_spaces[meta_model] = [
        uniform('{}-waiting-time'.format(meta_model), 90, 180, 30),
                       uniform('{}-ma-window-size'.format(meta_model), 50, 200, 50),
                       uniform('{}-th'.format(meta_model), .3, .5, .05),
                       loguniform('{}-l0'.format(meta_model), 1, 20),
                       loguniform('{}-l1'.format(meta_model), 1, 20),
                       uniform('{}-m'.format(meta_model), 1.1, np.e, .2),
                       ]
    return config_spaces
        

def config_space_to_configs(config_space, start=0, end=10):
    rng = np.random.RandomState(seed=0)
    configs = [config_space_sampler.sample(
        config_space, rng=rng) for i in range(end - start)]
    return configs[start:end]


if __name__ == '__main__':
    main()
