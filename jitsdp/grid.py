# coding=utf-8
import itertools


class Experiment():

    def __init__(self, experiment_config, seed_dataset_configs, models_configs):
        self.experiment_config = self.fix_experimnt_config(experiment_config)
        self.seed_dataset_configs = seed_dataset_configs
        self.models_configs = models_configs

    def fix_experimnt_config(self, experiment_config):
        self.rate_driven = experiment_config['rate_driven']
        self.meta_model = experiment_config['meta_model']
        new_experiment_config = dict(experiment_config)
        del new_experiment_config['rate_driven']
        del new_experiment_config['meta_model']
        return new_experiment_config

    @property
    def name(self):
        rate_driven = 'r' if self.rate_driven else ''
        model = self.experiment_config['model']
        train_data = 'cp' if self.experiment_config['cross_project'] else 'wp'
        return '{}{}-{}-{}'.format(rate_driven, self.meta_model, model, train_data)

    def to_shell(self, out):
        for seed_dataset_config in self.seed_dataset_configs:
            for models_config in self.models_configs:
                config = dict()
                config.update(self.experiment_config)
                config.update(seed_dataset_config)
                config.update(models_config)
                config = self.fix_rate_driven(config)
                config, entrypoint = self.fix_meta_model(config)
                config = self.add_experiment_name(config)
                params = ['--{} {}'.format(key, value)
                          for key, value in config.items()]
                params = ' '.join(params)
                out.write(
                    './{} {}\n'.format(entrypoint, params))

    def fix_rate_driven(self, config):
        config = dict(config)
        config['{}-rd'.format(self.meta_model)] = self.rate_driven
        return config

    def fix_meta_model(self, config):
        config = dict(config)
        meta_model_to_entrypoint = {
            'orb': 'baseline',
            'borb': 'jitsdp',
        }
        return config, meta_model_to_entrypoint[self.meta_model]

    def add_experiment_name(self, config):
        config = dict(config)
        config['experiment-name'] = self.name
        return config


def main():
    # experiments
    orb_rorb_grid = {
        'meta_model': ['orb'],
        'cross_project': [0, 1],
        'rate_driven': [0, 1],
        'model': ['hts'],
    }
    borb_rborb_grid = {
        'meta_model': ['borb'],
        'cross_project': [0, 1],
        'rate_driven': [0, 1],
        'model': ['ihf'],
    }
    rborb_grid = {
        'meta_model': ['borb'],
        'cross_project': [0, 1],
        'rate_driven': [1],
        'model': ['lr', 'mlp', 'nb', 'irf'],
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
    models_configs = {
        'hts': [{'hts-n-estimators': 1}],
        'ihf': [{'ihf-n-estimators': 1}],
        'lr': [],
        'mlp': [],
        'nb': [],
        'irf': [],
    }

    with open('jitsdp/dist/tuning.sh', mode='w') as out:
        for experiment in configs_to_experiments(experiment_configs, seed_dataset_configs, models_configs):
            experiment.to_shell(out)


def configs_to_experiments(experiment_configs, seed_dataset_configs, models_configs):
    for experiment_config in experiment_configs:
        model = experiment_config['model']
        experiment = Experiment(experiment_config=experiment_config,
                                seed_dataset_configs=seed_dataset_configs, models_configs=models_configs[model])
        yield experiment


def grid_to_configs(grid):
    keys = grid.keys()
    values_lists = grid.values()
    values_tuples = itertools.product(*values_lists)
    return list(map(lambda values_tuple: dict(zip(keys, list(values_tuple))), values_tuples))


if __name__ == '__main__':
    main()
