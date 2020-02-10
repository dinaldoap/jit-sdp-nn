from jitsdp.constants import DIR
from jitsdp.utils import mkdir

import pandas as pd
import re
from joblib import Memory


FEATURES = ['fix', 'ns', 'nd', 'nf', 'entrophy', 'la',
            'ld', 'lt', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp']
LABEL = 'contains_bug'

memory = Memory(location='logs', verbose=0)


@memory.cache
def make_stream(url):
    df_raw = download(url)
    df_preprocess = preprocess(df_raw)
    return prequential(df_preprocess)


def download(url):
    return pd.read_csv(url)


def preprocess(df_raw):
    preprocess_cols = ['commit_hash', 'author_date_unix_timestamp',
                       'fixes'] + FEATURES + [LABEL]
    df_preprocess = df_raw[preprocess_cols].copy()
    # timestamp
    df_preprocess = df_preprocess.rename(columns={'author_date_unix_timestamp': 'timestamp',
                                                  LABEL: 'target'})
    # filter rows with missing data
    df_preprocess = df_preprocess.dropna(subset=['fix'])
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
