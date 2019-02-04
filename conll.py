import re
from collections import defaultdict

def parse_dep_tree(lines, format):
    format = format or "ontonotes"

    tree = []
    for line in lines:
        fields = line.rstrip().split('\t')
        if format == "ontonotes":
            tree.append({'token': fields[1], 
                         'pos': fields[3],
                         'head': int(fields[6])-1,
                         'label': fields[7]})
        elif format == "ukwac":
            tree.append({'token': fields[0], 
                         'lemma': fields[1],
                         'pos': fields[2],
                         'head': int(fields[4])-1,
                         'label': fields[5]})
        else:
            raise ValueError("Unsupported format: %s" %format)

    # construct the list of tokens under each subtree
    n = len(tree)
    count = [0] * n
    for i in range(n):
        tree[i]['tokens'] = set()
        if tree[i]['head'] >= 0:
            count[tree[i]['head']] += 1
    # make a queue of tokens and process them one by one, leaves -> root
    tokens_to_process = [i for i in range(n) if count[i] == 0]
    while len(tokens_to_process) > 0:
        i = tokens_to_process.pop()
        if count[i] == 0:
            tree[i]['tokens'].add(i)
            if tree[i]['head'] >= 0:
                h = tree[i]['head']
                tree[h]['tokens'].update(tree[i]['tokens'])
                count[h] -= 1
                if count[h] == 0:
                    tokens_to_process.append(h)
    return tree

def load_trees(path):
    trees = []
    with open(path, 'rt') as f:
        lines = []
        for line in f:
            if re.match(r'\r?\n', line):
                trees.append(parse_dep_tree(lines))
                lines = []
            else:
                lines.append(line)
        if len(lines) > 0:
            trees.append(parse_dep_tree(lines))
    return trees
    
def find_syntactic_head(token_ids, dep):
    if not isinstance(token_ids, set):
        token_ids = set(token_ids)
    minimum_containing_subtree_head = None
    for h in range(len(dep)):
        if dep[h]['tokens'] == token_ids:
            return h
        else:
            if dep[h]['tokens'].issuperset(token_ids) and h in token_ids:
                if (minimum_containing_subtree_head is None or 
                    len(dep[minimum_containing_subtree_head]['tokens']) > len(dep[h]['tokens'])):
                    minimum_containing_subtree_head = h
    return minimum_containing_subtree_head

specificity = defaultdict(int, [('-NONE-', -5), ('IN', -4), ('RB', -3), 
                                ('PRP$', -2), ('PRP', -1), ('DT', -1),
                                ('VB', 1), ('VBN', 1), ('VBG', 1), ('NN', 1), ('NNS', 1), ('NNP', 1)])

def find_head(token_ids, dep):
    h = find_syntactic_head(token_ids, dep)
    if h is not None and dep[h]['pos'] in ['IN', 'RB']:
        children = [c for c in range(len(dep)) if dep[c]['head'] == h]
        if len(children) > 1:
#                 sys.stderr.write('multiple dependents of "%s": %s\n' 
#                                  %(dep[h]['token'], 
#                                    str([dep[c]['token'] for c in children])))
            c = max(children, key=lambda c:specificity[dep[c]['pos']])
            return c
        if len(children) > 0:
            return children[0]
#             else:
#                 if dep[h]['pos'] == 'IN':
#                     sys.stderr.write('preposition without dependent: %s\n' %dep[h]['token'])
#                     return '__INGORE__'
    return h
