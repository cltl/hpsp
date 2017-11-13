import ontonotes
from data import get_wordnet_pos, lemmatizer
from ontonotes import ontonotes_en, dev_docs
import os
import codecs
from collections import OrderedDict
import random

def iter_dependencies(doc):
    for sent_dep in doc.deps():
        for entry in sent_dep:
            word = entry['token'].lower()
            pos = entry['pos']
            lemma = lemmatizer.lemmatize(word, get_wordnet_pos(pos) or 'n')
            head_entry = sent_dep[entry['head']]
            head_word = head_entry['token'].lower()
            head_pos = head_entry['pos']
            head_lemma = lemmatizer.lemmatize(head_word, get_wordnet_pos(head_pos) or 'n')
            label = entry['label']
            yield head_lemma, label, lemma

def run():
    random.seed(329)
    print('Extracting from %s' %ontonotes_en)
    path1_train = 'output/dep_context.train.txt'
    path2_train = 'output/dep_context_labeled.train.txt'
    path1_dev = 'output/dep_context.dev.txt'
    path2_dev = 'output/dep_context_labeled.dev.txt'
    doc_count = 0
    with codecs.open(path1_train, 'w', 'utf-8') as f1_train, \
            codecs.open(path2_train, 'w', 'utf-8') as f2_train:
        tar_vocab = OrderedDict()
        for doc in ontonotes.list_docs(ontonotes_en):
            if os.path.relpath(doc.base_path, ontonotes_en) not in dev_docs:
                for head_lemma, label, lemma in iter_dependencies(doc):
                    f1_train.write('%s\t%s\n' %(head_lemma, lemma))
                    f2_train.write('%s_%s\t%s\n' %(head_lemma, label, lemma))
                    tar_vocab[lemma] = 1
            doc_count += 1
            if doc_count % 100 == 0: 
                print('Processed %d documents...' %doc_count)
    with codecs.open(path1_dev, 'w', 'utf-8') as f1_dev, \
            codecs.open(path2_dev, 'w', 'utf-8') as f2_dev:
        tar_vocab = list(tar_vocab)
        for doc in ontonotes.list_docs(ontonotes_en):
            if os.path.relpath(doc.base_path, ontonotes_en) in dev_docs:
                for head_lemma, label, lemma in iter_dependencies(doc):
                    neg = tar_vocab[random.randint(0, len(tar_vocab)-1)]
                    f1_dev.write('%s\t%s\t%s\n' %(head_lemma, lemma, neg))
                    f2_dev.write('%s_%s\t%s\t%s\n' %(head_lemma, label, lemma, neg))
    print("Read %d files" %doc_count)
    print("Written to %s, %s, %s, and %s" %(path1_train, path2_train, path1_dev, path2_dev))    

if __name__ == '__main__':
    run()