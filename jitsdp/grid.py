# coding=utf-8
from itertools import product


def main():
    general = {
        'seeds': [0, 1, 2, 3, 4],
        'datasets': ['brackets', 'camel', 'fabric8', 'jgroups', 'neutron', 'tomcat', 'broadleaf', 'nova', 'npm', 'spring-integration'],
        'cross_project': [0, 1]
    }
    lr = {
        'models': ['lr'],
        'lr_alpha': [.001, .01, .1],
        'lr_l1_ratio': [.15, .5, .85],
        'lr_n_epochs': [10, 30, 50],
    }
    lr.update(general)
    mlp = {
        'models': ['mlp'],
        'mlp_n_hidden_layers': [1, 2, 3, 4],
        'mlp_learning_rate': [.0001, .001, .01, .1, .3],
        'mlp_n_epochs': [10, 30, 50],
    }
    mlp.update(general)
    nb = {
        'models': ['nb'],
        'nb_n_updates': [10, 30, 50],
    }
    nb.update(general)
    irf = {
        'models': ['irf'],
        'irf_n_estimators': [50, 100, 150],
        'irf_criterion': ['entropy', 'gini'],
        'irf_max_depth': [3, 5, 7],
        'irf_max_features': [3, 5, 7],
        'irf_min_samples_leaf': [50, 100, 150],
        'irf_min_impurity_decrease': [.01, .02, .03]
    }
    irf.update(general)
    grids = [
        lr,
        mlp,
        nb,
        irf,
    ]
    with open('jitsdp/dist/grid.sh', mode='w') as out:
        for grid in grids:
            keys = grid.keys()
            values_lists = grid.values()
            for values_tuple in product(*values_lists):
                params = ['--{} {}'.format(key, values_tuple[i])
                          for i, key in enumerate(keys)]
                params = ' '.join(params)
                out.write(
                    './jitsdp run {}\n'.format(params))


if __name__ == '__main__':
    main()
