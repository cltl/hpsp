from lxml import etree
import re
from utils import call_for_all_children
from xml.sax.saxutils import unescape
from collections import defaultdict, Counter
from datetime import datetime
from nltk.corpus import wordnet as wn
import sys
import os
import numpy as np
import numpy
import pickle
from nltk.stem.wordnet import WordNetLemmatizer

class Indexer:
    
    def __init__(self):
        self.vocab = {}
        self.tokens = []
        self.sealed = False
        self.oov_index = None
        
    def index(self, token_or_tokens, ndmin=None):
        if isinstance(token_or_tokens, (list, tuple, set)):
            ret = []
            for v in token_or_tokens:
                ret.append(self.index(v))
            if ndmin is not None:
                ret = numpy.array(ret, ndmin=ndmin, dtype='int64')
            return ret
        return self._index_token(token_or_tokens)

    def _index_token(self, token):
        if token not in self.vocab:
            if self.sealed:
                if self.oov_index:
                    return self.oov_index
                else:
                    raise ValueError("Not allowed value: " + str(token))
            self.vocab[token] = len(self.tokens)
            self.tokens.append(token)
        return self.vocab[token]
    
    def token(self, index):
        return self.tokens[index]
    
    def __len__(self):
        return len(self.tokens)
    
    def seal(self, with_oov):
        if with_oov:
            self.oov_index = self.index('__OOV__')
        self.sealed = True
        
def mark_role(role_name):
    ''' Make a role different from a simple word '''
    # (Minh 2017-09-25) I don't implement this now because I'll have to go
    # through all usages of roles in the code. Fail to replace one and I'll
    # be in trouble. And the input doesn't contain A1, A2, etc. as a frame
    # element head anyway.
    return role_name

def read_embeddings(normalized=False):
    print("Reading embeddings...")
    with open('data/senna/words.lst') as f:
        word2id = dict((w, i) for i, w in enumerate(f.read().split('\n')))
    with open('data/senna/embeddings.txt') as f:
        embeddings = np.zeros((len(word2id), 50))
        for r, line in enumerate(f):
            for c, s in enumerate(line.strip().split(' ')):
                embeddings[r,c] = float(s)
            if (r+1) % 1000 == 0: sys.stdout.write("%5d " %(r+1))
            if (r+1) % 10000 == 0: sys.stdout.write('\n')
        if normalized:
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return word2id, embeddings


lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return ''
    