import logging
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='JIT-SDP: experiment execution')
    args = parser.parse_args()
    print('Args: {}'.format(args))
    logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    main()
