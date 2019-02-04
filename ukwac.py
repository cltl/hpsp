from conll import parse_dep_tree
import gzip
from tqdm import tqdm
from glob import glob
import re
from time import time
import sys


def iter_sentences(lines):
    buffer = []
    for line in lines:
        line = line.rstrip().decode('utf-8', 'ignore')
        if line == '<s>' or line.startswith('<text id='):
            buffer = []
        elif line in ('</s>', '</text>'):
            yield buffer
        else:
            buffer.append(line)

def count_lines_in_gz_fast(path):
    def blocks(f, size=2**20):
        while True:
            b = f.read(size)
            if not b: break
            yield b

    with gzip.open(path, 'r') as f:
        return sum(bl.count(b"\n") for bl in blocks(f))

def iter_lines_counting(path, name):
    print('%s: Counting lines in "%s"' %(name, path))
    total_lines = count_lines_in_gz_fast(path)
    
    start_sec = time()
    with gzip.open(path, 'rb') as f_inp:
        for num_line, line in enumerate(f_inp):
            yield line
            if (num_line+1) % 1000000 == 0:
                elapsed_min = (time() - start_sec) / 60.0
                print('%s: %d lines (%.1f%%), elapsed: %.1f min' 
                      %(name, num_line+1, 100*(num_line+1)/total_lines, elapsed_min))
                sys.stdout.flush()

def iter_dep_trees(paths):
    for path in paths:
        print('Reading "%s"' %path)
        with gzip.open(path, 'rb') as f:
            for dep_tree in iter_dep_trees_from_file(tqdm(f, unit='lines')):
                yield dep_tree


def iter_dep_trees_from_file(f):
    for sent_no, sent_lines in enumerate(iter_sentences(f)):
        try:
            yield parse_dep_tree(sent_lines, 'ukwac')
        except:
            print('Error occurred at sentence %d. Skipped.' %(sent_no+1))
            print(sent_lines)


def iter_triples(dep_trees):
    for sent_deps in dep_trees:
        dobjs = [d for i, d in enumerate(sent_deps) 
                 if d['label'] == 'OBJ' and is_valid_noun(d)]
        for dobj in dobjs:
            verb_index = dobj['head']
            verb = sent_deps[verb_index]
            if is_valid_verb(verb):
                sbjs = [d for i, d in enumerate(sent_deps) 
                        if d['head'] == verb_index and d['label'] == 'SBJ' and is_valid_noun(d)]
                for sbj in sbjs:
                    yield (normalize(sbj), normalize(verb), normalize(dobj))


def is_valid_noun(dep):
    # "only keep those forms that contain alphabetic characters" (van de Cruys, 2014)
    return dep['pos'] in ('NN', 'NNS') and re.match('[a-zA-Z]+$', dep['token'])


def is_valid_verb(dep):
    # "only keep those forms that contain alphabetic characters" (van de Cruys, 2014)
    return dep['pos'].startswith('V') and re.match('[a-zA-Z]+$', dep['token'])


def normalize(dep):
    # "All words are converted to lowercase. We use the lemmatized forms" (van de Cruys, 2014)
    return dep['lemma'].lower()