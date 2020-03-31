from jitsdp.constants import DIR
from jitsdp.utils import mkdir

from joblib import Memory
import pandas as pd
import re


FEATURES = ['fix', 'ns', 'nd', 'nf', 'entrophy', 'la',
            'ld', 'lt', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp']
LABEL = 'contains_bug'

memory = Memory(location='data', verbose=0)


@memory.cache
def make_stream(url):
    dataset = re.search('(?P<dataset>\\w+)\\.csv', url)
    dataset = dataset.group('dataset')
    df_raw = download(url)
    if dataset in ['brackets', 'camel', 'fabric8', 'jgroups', 'neutron', 'tomcat']:
        df_preprocess = preprocess(df_raw)
    elif dataset in ['broadleaf', 'nova', 'npm', 'spring-integration']:
        df_preprocess = preprocess_daystofix(df_raw)
    else:
        raise NotImplementedError('Dataset not supported.')
    return prequential(df_preprocess)


def download(url):
    return pd.read_csv(url, skipinitialspace=True)


def preprocess(df_raw):
    preprocess_cols = ['commit_hash', 'author_date_unix_timestamp',
                       'fixes'] + FEATURES + [LABEL]
    df_preprocess = df_raw[preprocess_cols].copy()
    # timestamp
    df_preprocess = df_preprocess.rename(columns={'author_date_unix_timestamp': 'timestamp',
                                                  LABEL: 'target'})
    # filter rows with missing data
    df_preprocess = df_preprocess.dropna(subset=['fix'])
    df_preprocess['fix'] = df_preprocess['fix'].astype('int')
    # timeline order
    df_preprocess = df_preprocess[::-1]
    df_preprocess = df_preprocess.reset_index(drop=True)
    # contains_bug
    df_preprocess['target'] = df_preprocess['target'].astype('int')
    # fixes
    df_preprocess['commit_hash_fix'] = df_preprocess['fixes'].dropna().apply(
        lambda x: re.findall('\\b\\w+\\b', x)[0])
    df_fix = df_preprocess[['commit_hash', 'timestamp']
                           ].set_index('commit_hash')
    return df_preprocess.join(df_fix, on='commit_hash_fix', how='left', rsuffix='_fix')


def preprocess_daystofix(df_raw):
    label = 'containsbug'
    preprocess_cols = ['timestamp',
                       'daystofix'] + FEATURES + [label]
    df_preprocess = df_raw[preprocess_cols].copy()
    # timestamp
    df_preprocess = df_preprocess.rename(columns={label: 'target'})
    # contains_bug
    df_preprocess['target'] = df_preprocess['target'].astype('int')
    # zero to nan
    df_preprocess['daystofix'] = df_preprocess['daystofix'].apply(
        lambda x: None if x == 0. else x)
    # fixes
    df_preprocess['timestamp_fix'] = df_preprocess['timestamp'] + \
        df_preprocess['daystofix'] * 24 * 60 * 60
    return df_preprocess


def prequential(df_preprocess):
    prequential_cols = ['timestamp', 'timestamp_fix'] + \
        FEATURES + ['target']
    df_prequential = df_preprocess[prequential_cols].copy()
    df_prequential['timestep'] = range(len(df_prequential))
    return df_prequential


def save_results(results, dir=DIR):
    mkdir(dir)
    results.to_pickle(dir / 'results.pickle')


def load_results(dir=DIR):
    return pd.read_pickle(dir / 'results.pickle')
