import codecs
from collections import Counter, defaultdict
import random

def get_bin(count):
    if count < 100:
        return round(count, -1)
    else: 
        return round(count, -2)

if __name__ == '__main__':
    tar_counter = Counter()
    with codecs.open('output/dep_context_labeled.train.txt', 'r', 'utf-8') as f:
        for line in f:
            _, target = line.strip().split('\t')
            tar_counter[target] += 1
    freq_bins = defaultdict(list)
    for tar, count in tar_counter.items():
        freq_bins[get_bin(count)].append(tar)
    print('Frequency bins: ')
    for key, tars in sorted(freq_bins.items()):
        print('%d\t%d' %(key, len(tars)))

    out_path = 'output/dep_context_labeled.dev-fixed.txt'
    with codecs.open('output/dep_context_labeled.dev.txt', 'r', 'utf-8') as f, \
            codecs.open(out_path, 'w', 'utf-8') as f2:
        for line in f:
            contex, target_pos, _ = line.strip().split('\t')
            candidates = freq_bins[get_bin(tar_counter[target_pos])]
            if len(candidates) >= 10:
                target_neg = candidates[random.randint(0, len(candidates)-1)]
                f2.write('%s\t%s\t%s\n' %(contex, target_pos, target_neg))
    print('Written to %s' %out_path)