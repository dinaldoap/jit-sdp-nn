from jitsdp.evaluation import prequential

import logging
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='JIT-SDP: experiment execution')
    args = parser.parse_args()
    print('Args: {}'.format(args))
    logging.getLogger('').handlers = []
    logging.basicConfig(filename='logs/jitsdp.log', filemode='w', level=logging.DEBUG)
    prequential()


if __name__ == '__main__':
    main()
