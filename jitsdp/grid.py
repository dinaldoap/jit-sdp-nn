from itertools import product

def main():
    general = {
        'seeds': [0, 1, 2, 3, 4],
        'datasets': ['brackets', 'camel', 'fabric8', 'jgroups', 'neutron', 'tomcat', 'broadleaf', 'nova', 'npm', 'spring-integration'],
    }
    lr = {
        'models': ['lr'],
        'n_epochs': [10, 30, 50],
    }
    lr.update(general)

    grids = [
        lr,
    ]
    with open('jitsdp/dist/grid.sh', mode='w') as out:
        for grid in grids:
            keys = grid.keys()
            values_lists = grid.values()
            for values_tuple in product(*values_lists):
                params = ['--{} {}'.format(key, values_tuple[i]) for i, key in enumerate(keys)]
                params = ' '.join(params)
                out.write('./borb run --f_folds 0. --orb 1 {}\n'.format(params))


if __name__ == '__main__':
    main()