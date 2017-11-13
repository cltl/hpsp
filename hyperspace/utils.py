import codecs
import numpy as np
import sys

def read_train_dataset(path, cxt_indexer, tar_indexer):
    sys.stderr.write('Reading a dataset from %s ... ' %path)
    x, y = [], []
    with codecs.open(path, 'r', 'utf-8') as f:
        for line in f:
            context, target = line.strip().split('\t')
            x.append(cxt_indexer.index(context))
            y.append(tar_indexer.index(target))
    sys.stderr.write('Done.\n')
    return np.array(x), np.array(y)

def read_dev_dataset(path, cxt_indexer, tar_indexer):
    sys.stderr.write('Reading a dataset from %s ... ' %path)
    x, y_pos, y_neg = [], [], []
    with codecs.open(path, 'r', 'utf-8') as f:
        for line in f:
            context, target_pos, target_neg = line.strip().split('\t')
            x.append(cxt_indexer.index(context))
            y_pos.append(tar_indexer.index(target_pos))
            y_neg.append(tar_indexer.index(target_neg))
    sys.stderr.write('Done.\n')
    return np.array(x), np.array(y_pos), np.array(y_neg)
