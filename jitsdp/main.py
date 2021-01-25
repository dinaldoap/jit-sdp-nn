# coding=utf-8
# torch and sklearn imported first to avoid bug
# see https://github.com/pytorch/pytorch/issues/2575#issuecomment-523657178
from jitsdp.evaluation import run
from jitsdp.utils import setup_and_run, int_or_none
from jitsdp import baseline, tuning, testing, report

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description='JIT-SDP: experiment execution tool')
    subparsers = parser.add_subparsers(
        title='meta-model or script generator', dest='meta_model', required=True)
    add_arguments(subparsers.add_parser(
        name='borb', help='Run Batch Oversampling Rate Boosting meta-model'))
    baseline.add_arguments(subparsers.add_parser(
        name='orb', help='Run Oversampling Rate Boosting meta-model'))
    tuning.add_arguments(subparsers.add_parser(
        name='tuning', help='Generate hyperparameter tuning script'), 'logs/tuning.sh')
    testing.add_arguments(subparsers.add_parser(
        name='testing', help='Generate testing script'), 'logs/testing.sh')
    report.add_arguments(subparsers.add_parser(
        name='report', help='Generate report'), 'logs')

    args = parser.parse_args()
    config = dict(vars(args))
    meta_model_generator = config['meta_model']
    if meta_model_generator == 'borb':
        return setup_and_run(config, run)
    elif meta_model_generator == 'orb':
        return setup_and_run(config, baseline.run)
    elif meta_model_generator == 'tuning':
        return tuning.generate(config)
    elif meta_model_generator == 'testing':
        return testing.generate(config)
    elif meta_model_generator == 'report':
        return report.generate(config)
    else:
        raise ValueError('meta-model and script generator not supported.')


def add_arguments(parser):
    parser.add_argument('--experiment-name',   type=str,
                        help='Experiment name (default: None). None means default behavior of MLflow', default=None)
    parser.add_argument('--start',   type=int,
                        help='First commit to be used for testing (default: 0).',    default=0)
    parser.add_argument('--end',   type=int_or_none,
                        help='Last commit to be used for testing (default: 5000). None means all commits.',  default=5000)
    parser.add_argument('--borb-waiting-time',   type=int,
                        help='Number of days to wait before labeling the commit as clean (default: 90).',    default=90)
    parser.add_argument('--borb-ma-window-size',   type=int,
                        help='Number of recent commits to use to calculate the moving average of the model\'s output (default: 100).',    default=100)
    parser.add_argument('--borb-pull-request-size',   type=int,
                        help='Number of commits to wait before retraining (default: 50).',    default=50)
    parser.add_argument('--borb-sample-size',   type=int_or_none,
                        help='Sample size selected from the training data in each iteration (default: 1000). None means bootstrap.',    default=1000)
    parser.add_argument('--borb-th',   type=float,
                        help='Expected value for the moving average of the model\'s output (default: .4).',  default=.4)
    parser.add_argument('--borb-l0',   type=float,
                        help='No description. (default: 10.).',  default=10.)
    parser.add_argument('--borb-l1',   type=float,
                        help='No description. (default: 12.).',  default=12.)
    parser.add_argument('--borb-m',   type=float,
                        help='No description. (default: 1.5).',  default=1.5)
    parser.add_argument('--borb',   type=int,
                        help='Whether must use oversampling rate boosting to balance output proportions (default: 1).', default=1, choices=[0, 1])
    parser.add_argument('--cross-project',   type=int,
                        help='Whether must use cross-project data (default: 0).', default=0, choices=[0, 1])
    parser.add_argument('--seed',   type=int,
                        help='Seed of random state (default: 273243676114050081847384039665342324335).',    default=273243676114050081847384039665342324335)
    parser.add_argument('--dataset',   type=str, help='Dataset to run the experiment. (default: brackets).',
                        default='brackets', choices=['brackets', 'camel', 'fabric8', 'jgroups', 'neutron', 'tomcat', 'broadleaf', 'nova', 'npm', 'spring-integration'])
    parser.add_argument('--model',   type=str,
                        help='Which model must use as the base learner (default: irf).', default='mlp', choices=['ihf', 'lr', 'mlp', 'nb', 'irf'])
    parser.add_argument('--ihf-n-estimators',   type=int,
                        help='The number of hoeffding trees (default: 1).',  default=1)
    parser.add_argument('--ihf-grace-period',   type=int,
                        help='Number of instances a leaf should observe between split attempts (default: 200).',  default=200)
    parser.add_argument('--ihf-split-criterion',   type=str, help='Split criterion to use (default: info_gain).',
                        default='info_gain', choices=['gini', 'info_gain', 'hellinger'])
    parser.add_argument('--ihf-split-confidence',   type=float,
                        help='Allowed error in split decision, a value closer to 0 takes longer to decid (default: .0000001).',  default=.0000001)
    parser.add_argument('--ihf-tie-threshold',   type=float,
                        help='Threshold below which a split will be forced to break ties (default: .05).',  default=.05)
    parser.add_argument('--ihf-remove-poor-atts',   type=int,
                        help='Whether must disable poor attributes (default: 0).',
                        default=0, choices=[0, 1])
    parser.add_argument('--ihf-no-preprune',   type=int,
                        help='Whether must disable pre-pruning (default: 0).',
                        default=0, choices=[0, 1])
    parser.add_argument('--ihf-leaf-prediction',   type=str, help='Prediction mechanism used at leafs. (default: nba).',
                        default='nba', choices=['mc', 'nb', 'nba'])
    parser.add_argument('--ihf-n-updates',   type=int,
                        help='Number of updates performed by each hoeffding tree with the same sample (default: 1).',    default=1)
    parser.add_argument('--lr-alpha',   type=float,
                        help='Constant that multiplies the regularization term. Also used to compute the learning rate (default: .1).',  default=.1)
    parser.add_argument('--lr-l1-ratio',   type=float,
                        help='The Elastic Net mixing parameter (default: .15).',  default=.15)
    parser.add_argument('--lr-n-epochs',   type=int,
                        help='Number of epochs performed by the training (default: 1).',    default=1)
    parser.add_argument('--lr-batch-size',   type=int,
                        help='Number of commits included in each batch (default: 256).',    default=256)
    parser.add_argument('--lr-log-transformation',   type=int,
                        help='Whether must use log transformation (default: 0).',
                        default=0, choices=[0, 1])
    parser.add_argument('--mlp-n-hidden-layers',   type=int,
                        help='Number of hidden layers (default: 1).',    default=1)
    parser.add_argument('--mlp-hidden-layers-size',   type=int,
                        help='Hidden layers size (default: 7).',    default=7)
    parser.add_argument('--mlp-learning-rate',   type=float,
                        help='Learning rate (default: .001).',  default=.001)
    parser.add_argument('--mlp-n-epochs',   type=int,
                        help='Number of epochs performed by the training (default: 1).',    default=1)
    parser.add_argument('--mlp-dropout-input-layer',   type=float,
                        help='Dropout probability of the input layer (default: .2).',    default=.2)
    parser.add_argument('--mlp-dropout-hidden-layers',   type=float,
                        help='Dropout probability of the hidden layers (default: .5).',    default=.5)
    parser.add_argument('--mlp-batch-size',   type=int,
                        help='Number of commits included in each batch (default: 256).',    default=256)
    parser.add_argument('--mlp-log-transformation',   type=int,
                        help='Whether must use log transformation (default: 0).',
                        default=0, choices=[0, 1])
    parser.add_argument('--nb-n-updates',   type=int,
                        help='Number of updates performed by the training (default: 1).',    default=1)
    parser.add_argument('--irf-n-estimators',   type=int,
                        help='The number of trees in the forest (default: 1).',    default=1)
    parser.add_argument('--irf-criterion',   type=str,
                        help='The function to measure the quality of a split (default: \'entropy\').', default='entropy', choices=['entropy', 'gini'])
    parser.add_argument('--irf-max-depth',   type=int,
                        help='The maximum depth of the tree (default: unlimited).', default=None)
    parser.add_argument('--irf-max-features',   type=int,
                        help='The number of features to consider when looking for the best split (default: 3).', default=3)
    parser.add_argument('--irf-min-samples-leaf',   type=int,
                        help='The minimum number of samples required to be at a leaf node (default: 200).', default=200)
    parser.add_argument('--irf-min-impurity-decrease',   type=float,
                        help='A node will be split if this split induces a decrease of the impurity greater than or equal to this value (default: .0).', default=.0)
    parser.add_argument('--track-time',   type=int,
                        help='Whether must track time. (default: 0).',  default=0)
    parser.add_argument('--track-forest',   type=int,
                        help='Whether must track forest state. (default: 0).',  default=0)
    parser.add_argument('--f-val',   type=float,
                        help='Fraction of labeled data to be used for validation. (default: .0).',  default=.0)
    parser.add_argument('--ensemble-size',   type=int,
                        help='Number of models in the ensemble (default: 1).',    default=1)
    parser.add_argument('--threshold',   type=int,
                        help='Whether must tune threshold to balance output proportions (default: 0).', default=0, choices=[0, 1])
    parser.add_argument('--uncertainty',   type=int,
                        help='Whether must use decreasing uncertainty about normal commit labels inside verification latency (default: 0).', default=0, choices=[0, 1])
    parser.add_argument('--incremental',   type=int,
                        help='Whether must do incremental training along the stream (default: 0).', default=0, choices=[0, 1])


if __name__ == '__main__':
    sys.exit(main())
