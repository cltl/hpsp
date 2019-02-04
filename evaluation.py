import math
import random
from collections import OrderedDict
import pandas as pd


def compute_frequency_band_mapping(series):
    counts = series.value_counts()
    n_tenth = int(math.ceil(len(counts)/10))
    bands = [counts.iloc[i*n_tenth:(i+1)*n_tenth].index.values for i in range(10)]
    assert len(counts) == sum(len(band) for band in bands)
    return {val:band for band in bands for val in band}


def create_test_set_from_positive_examples_same_freq(df, random_seed=None):
    ''' Complete a dataset of positive triples by adding negative one
    sampled such that the negative subject and object is of the same
    frequence band as the negative ones.

    Input: df: Pandas DataFrame with three columns: `sbj`, `verb`, and `dobj`
    '''
    pos_examples_as_set = set(tuple(row) for row in df[['sbj', 'verb', 'dobj']].values)
    dobj2band = compute_frequency_band_mapping(df['dobj'])
    sbj2band = compute_frequency_band_mapping(df['sbj'])

    rng = random.Random(random_seed)
    test_examples = []
    for pos_sbj, verb, pos_dobj in pos_examples_as_set:
        neg_sbj, neg_dobj = pos_sbj, pos_dobj
        while (neg_sbj, verb, neg_dobj) in pos_examples_as_set:
            neg_sbj = rng.choice(sbj2band[pos_sbj])
            neg_dobj = rng.choice(dobj2band[pos_dobj])
        test_examples.append(OrderedDict([('verb', verb), 
                                          ('pos_sbj', pos_sbj), ('pos_dobj', pos_dobj),
                                          ('neg_sbj', neg_sbj), ('neg_dobj', neg_dobj)]))
    return pd.DataFrame(test_examples)


def create_test_set_from_positive_examples_unigram_freq(df, random_seed=None):
    pos_examples_as_set = set(tuple(row) for row in df[['sbj', 'verb', 'dobj']].values)
    rng = random.Random(random_seed)
    test_examples = []
    for pos_sbj, verb, pos_dobj in pos_examples_as_set:
        neg_sbj, neg_dobj = pos_sbj, pos_dobj
        while (neg_sbj, verb, neg_dobj) in pos_examples_as_set:
            neg_sbj = rng.choice(df.sbj.values)
            neg_dobj = rng.choice(df.dobj.values)
        test_examples.append(OrderedDict([('verb', verb), 
                                          ('pos_sbj', pos_sbj), ('pos_dobj', pos_dobj),
                                          ('neg_sbj', neg_sbj), ('neg_dobj', neg_dobj)]))
    return pd.DataFrame(test_examples)