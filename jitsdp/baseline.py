from jitsdp import metrics as met
from jitsdp.data import make_stream, make_stream_others, save_results, DATASETS, FEATURES
from jitsdp.orb import ORB
from jitsdp.pipeline import MultiflowBaseEstimator, MultiflowForest, set_seed
from jitsdp.report import report
from jitsdp.utils import int_or_none

import mlflow
import pandas as pd
from skmultiflow.data import DataStream
from skmultiflow.meta import OzaBaggingClassifier
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.bayes import NaiveBayes


def add_arguments(parser):
    parser.add_argument('--experiment-name',   type=str,
                        help='Experiment name (default: None). None means default behavior of MLflow', default=None)
    parser.add_argument('--start',   type=int,
                        help='First commit to be used for testing (default: 0).',    default=0)
    parser.add_argument('--end',   type=int_or_none,
                        help='Last commit to be used for testing (default: None). None means all commits.',  default=5000)
    parser.add_argument('--orb-waiting-time',   type=int,
                        help='Number of days to wait before labeling the commit as clean (default: 90).',    default=90)
    parser.add_argument('--orb-ma-window-size',   type=int,
                        help='The number of predictions or instances used for calculating moving average (default: 100).',  default=100)
    parser.add_argument('--orb-th',   type=float,
                        help='Expected value for the moving average of the model\'s output (default: .4).',  default=.4)
    parser.add_argument('--orb-l0',   type=float,
                        help='No description (default: 10.).',  default=10.)
    parser.add_argument('--orb-l1',   type=float,
                        help='No description (default: 12.).',  default=12.)
    parser.add_argument('--orb-m',   type=float,
                        help='No description (default: 1.5).',  default=1.5)
    parser.add_argument('--orb-decay-factor',   type=float,
                        help='Decay factor for calculating class proportions in training data (default: .99).',  default=.99)
    parser.add_argument('--orb-n',   type=int,
                        help='The number of clean commits that activate the noise filter (default: 3).',  default=3)
    parser.add_argument('--cross-project',   type=int,
                        help='Whether must use cross-project data (default: 0).', default=0, choices=[0, 1])
    parser.add_argument('--noise',   type=int,
                        help='Whether must keep noisy instances (default: 0).',
                        default=0, choices=[0, 1])
    parser.add_argument('--order',   type=int,
                        help='Whether must keep the order of the events (default: 0).',
                        default=0, choices=[0, 1])
    parser.add_argument('--seed',   type=int,
                        help='Seed of random state (default: 273243676114050081847384039665342324335).',    default=273243676114050081847384039665342324335)
    parser.add_argument('--dataset',   type=str, help='Dataset to run the experiment. (default: brackets).',
                        default='brackets', choices=['brackets', 'camel', 'fabric8', 'jgroups', 'neutron', 'tomcat', 'broadleaf', 'nova', 'npm', 'spring-integration'])
    parser.add_argument('--model',   type=str,
                        help='Which models must use as the base learner (default: oht).', default='oht', choices=['nb', 'oht'])
    parser.add_argument('--oht-n-estimators',   type=int,
                        help='The number of hoeffding trees (default: 1).',  default=1)
    parser.add_argument('--oht-grace-period',   type=int,
                        help='Number of instances a leaf should observe between split attempts (default: 200).',  default=200)
    parser.add_argument('--oht-split-criterion',   type=str, help='Split criterion to use (default: info_gain).',
                        default='info_gain', choices=['gini', 'info_gain', 'hellinger'])
    parser.add_argument('--oht-split-confidence',   type=float,
                        help='Allowed error in split decision, a value closer to 0 takes longer to decid (default: .0000001).',  default=.0000001)
    parser.add_argument('--oht-tie-threshold',   type=float,
                        help='Threshold below which a split will be forced to break ties (default: .05).',  default=.05)
    parser.add_argument('--oht-remove-poor-atts',   type=int,
                        help='Whether must disable poor attributes (default: 0).',
                        default=0, choices=[0, 1])
    parser.add_argument('--oht-no-preprune',   type=int,
                        help='Whether must disable pre-pruning (default: 0).',
                        default=0, choices=[0, 1])
    parser.add_argument('--oht-leaf-prediction',   type=str, help='Prediction mechanism used at leafs. (default: nba).',
                        default='nba', choices=['mc', 'nb', 'nba'])
    parser.add_argument('--track-time',   type=int,
                        help='Whether must track time. (default: 0).',  default=0)
    parser.add_argument('--track-forest',   type=int,
                        help='Whether must track forest state (default: 0)',  default=0)
    parser.add_argument('--track-orb',   type=int,
                        help='Whether must track ORB state (default: 0)',  default=0)


def run(config):
    mlflow.log_params(config)
    set_seed(config)
    dataset = config['dataset']
    # stream with commit order
    df_commit = make_stream(dataset)
    # stream with labeling order
    end = len(df_commit) if config['end'] is None else config['end']
    df_test = df_commit[:end].copy()
    df_train = df_commit[:end].copy()
    if config['cross_project']:
        df_train = merge_others(df_train, dataset)
    df_train = extract_events(df_train, config['orb_waiting_time'])
    if not config['noise']:
        df_train = remove_noise(df_train, config['orb_n'])
    if not config['order']:
        df_train = balance_events(df_train)

    test_steps = calculate_steps(
        df_test['timestamp'], df_train['timestamp_event'], right=False)
    train_steps = calculate_steps(
        df_train['timestamp_event'], df_test['timestamp'], right=True)
    train_steps = train_steps.to_list()

    train_stream = DataStream(df_train[FEATURES], y=df_train[['target']])
    base_learner = create_classifier(config)
    model = ORB(features=FEATURES,
                decay_factor=config['orb_decay_factor'],
                ma_window_size=config['orb_ma_window_size'],
                th=config['orb_th'],
                l0=config['orb_l0'],
                l1=config['orb_l1'],
                m=config['orb_m'],
                base_learner=base_learner
                )
    target_prediction = None
    train_first = len(test_steps) < len(train_steps)
    current_test = 0
    for test_index, test_step in test_steps.items():
        # train
        if train_first:
            train_step = train_steps.pop(0)
            X_train, y_train = train_stream.next_sample(train_step)
            model.train(
                X_train, y_train, track_orb=config['track_orb'])
        else:
            train_first = True
        # test
        df_batch_test = df_test[current_test:current_test + test_step]
        current_test += test_step
        target_prediction_test = model.predict(
            df_batch_test, track_time=config['track_time'], track_forest=config['track_forest'])
        target_prediction = pd.concat(
            [target_prediction, target_prediction_test])

    target_prediction = target_prediction[config['start']:end]
    target_prediction = target_prediction.reset_index(drop=True)

    results = met.prequential_metrics(target_prediction, .99, config['orb_th'])
    report(config, results)


def extract_events(df_commit, waiting_time):
    seconds_by_day = 24 * 60 * 60
    # seconds
    verification_latency = waiting_time * seconds_by_day
    # cleaned
    df_clean = df_commit[df_commit['target'] == 0]
    df_cleaned = df_clean.copy()
    df_cleaned['timestamp_event'] = df_cleaned['timestamp'] + \
        verification_latency
    # bugged
    df_bug = df_commit[df_commit['target'] == 1]
    df_bugged = df_bug.copy()
    df_bugged['timestamp_event'] = df_bugged['timestamp_fix'].astype(int)
    # bug cleaned
    df_bug_cleaned = df_bug.copy()
    waited_time = df_bug_cleaned['timestamp_fix'] - df_bug_cleaned['timestamp']
    df_bug_cleaned = df_bug_cleaned[waited_time >= verification_latency]
    df_bug_cleaned['target'] = 0
    df_bug_cleaned['timestamp_event'] = df_bug_cleaned['timestamp'] + \
        verification_latency
    # events
    df_events = pd.concat([df_cleaned, df_bugged, df_bug_cleaned])
    df_events = df_events.sort_values('timestamp_event', kind='mergesort')
    df_events = df_events[['timestamp_event'] + FEATURES + ['target']]
    return df_events


def remove_noise(df_events, orb_n):
    grouped_target = df_events.groupby(FEATURES)['target']
    cumsum = grouped_target.cumsum()
    cumcount = grouped_target.cumcount()
    noise = cumcount - cumsum >= orb_n
    noise = noise & (df_events['target'] == 1)
    return df_events[~noise]


def balance_events(df_events):
    bug_pool = []
    df_balanced = []
    for row in df_events.itertuples(index=False):
        if row.target == 1:
            bug_pool.append(row)

        if row.target == 0:
            df_balanced.append(row)
            if len(bug_pool) > 0:
                bug = bug_pool.pop(0)
                bug = bug._replace(timestamp_event=row.timestamp_event)
                df_balanced.append(bug)

    df_balanced = pd.DataFrame(df_balanced)
    return df_balanced


def calculate_steps(data, bins, right):
    min_max = pd.concat([data[:1] - int(right), data[-1:] + int(not right)])
    internal_bins = bins[(min_max.min() < bins) & (bins < min_max.max())]
    full_bins = pd.concat([min_max[:1], internal_bins, min_max[-1:]])
    full_bins = full_bins.drop_duplicates()
    steps = pd.cut(data, bins=full_bins, right=right, include_lowest=True)
    steps = steps.value_counts(sort=False)
    steps = steps[steps > 0]
    return steps


def merge_others(data, dataset):
    df_others = make_stream_others(dataset)
    last_timestamp = data['timestamp'].max()
    df_others = df_others[df_others['timestamp'] <= last_timestamp]
    return pd.concat([data, df_others])


def create_classifier(config):
    map_fn = {
        'nb': create_nb_model,
        'oht': create_oht_model,
    }
    fn_create_model = map_fn[config['model']]
    return fn_create_model(config)


def create_nb_model(config):
    return MultiflowBaseEstimator(NaiveBayes())


def create_oht_model(config):
    base_estimator = HoeffdingTreeClassifier(
        grace_period=config['oht_grace_period'],
        split_criterion=config['oht_split_criterion'],
        split_confidence=config['oht_split_confidence'],
        tie_threshold=config['oht_tie_threshold'],
        remove_poor_atts=config['oht_remove_poor_atts'],
        no_preprune=config['oht_no_preprune'],
        leaf_prediction=config['oht_leaf_prediction'])
    n_estimators = config['oht_n_estimators']
    return MultiflowForest(OzaBaggingClassifier(
        base_estimator=base_estimator, n_estimators=n_estimators))
