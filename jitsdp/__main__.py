from jitsdp.evaluation import prequential

import logging
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='JIT-SDP: experiment execution')
    parser.add_argument('--epochs',   type=int, help='Number of epochs performed by the training (default: 1).',    default=1)
    parser.add_argument('--folds',   type=float, help='Fraction of folds to be used by the evaluation. A minimum of two folds is always used despite this parameter. (default: 0).',    default=0)    
    config = parser.parse_args()
    print('Configuration: {}'.format(config))
    logging.getLogger('').handlers = []
    logging.basicConfig(filename='logs/jitsdp.log',
                        filemode='w', level=logging.DEBUG)
    prequential(vars(config))


if __name__ == '__main__':
    main()
