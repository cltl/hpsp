from nltk.tree import ParentedTree
import os

root_dir = 'data/ptb'
# assert os.path.exists(root_dir), 'please link/put PENN TreeBank into data directory'

def _assign_token_ids(root, n):
    assert not isinstance(root, str)
    if len(root) == 1 and isinstance(root[0], str):
        if root.label() == '-NONE-':
            root.token_id = None
        else:
            root.token_id = n
            n += 1
    else:
        for child in root:
            n = _assign_token_ids(child, n)
    return n

def _find_terminals(root):
    assert not isinstance(root, str)
    target_list = []
    if len(root) == 1 and isinstance(root[0], str):
        target_list.append(root)
    else:
        for child in root:
            target_list.extend(_find_terminals(child))
    root.terminals = target_list
    return target_list
    
def _tree_from_string(s):
    root = ParentedTree.fromstring(s)
    _find_terminals(root)
    _assign_token_ids(root, 0)
    return root

def _iter_syntax_trees(path):
    with open(path, 'rt') as f:
        parenthese_count = 0
        buf = []
        line = f.readline()
        while line != '':
            if parenthese_count == 0:
                s = ' '.join(buf).strip()
                if s: yield s
                del buf[:]
            buf.append(line)
            parenthese_count += sum(1 for c in line if c == '(')
            parenthese_count -= sum(1 for c in line if c == ')')
            line = f.readline()
        s = ' '.join(buf).strip()
        if s: yield s
    
def load_trees(path):
    trees = []
    for s in _iter_syntax_trees(path):
        root = _tree_from_string(s)
        trees.append(root)
    return trees
