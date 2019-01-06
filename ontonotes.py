import os
import re
from collections import defaultdict
import sys
import ptb
import conll
from nltk.corpus.reader.wordnet import WordNetError
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import codecs
from data import Frame, FrameElement, Document, get_wordnet_pos, lemmatizer
from lxml import etree
from conll import find_syntactic_head
import csv
import random

ontonotes_root = 'data/ontonotes-release-5.0'
ontonotes_en = os.path.join(ontonotes_root, 'data/files/data/english/annotations')
assert os.path.exists(ontonotes_root), 'please link/put ontonotes into data directory'

def load_coref(content, dep_content=None):
    chains = defaultdict(list)
    sent = 0
    part_no = ''
    for line in content.split('\n'):
        m_part_no = re.match(r'<TEXT\s+PARTNO="(\d+)">', line)
        if m_part_no:
            part_no = m_part_no.group(1)
        if line.strip() and not re.match(r'</?DOC|</?TEXT', line):
            tokens = []
            brackets = []
            curr_index = 0
            parts = re.findall(r'<COREF\s[^>]+>|</COREF\s*>|[^\s<>]+', line)
            for s in parts:
                m = re.match(r'<COREF\s+ID="([^"]+)"', s)
                if m:
                    brackets.append((m.group(1), curr_index))
                elif re.match(r'</COREF', s):
                    refid, start_index = brackets.pop()
                    chains['p%s_%s' %(part_no, refid)].append((sent, start_index, curr_index))
                else:
                    s = (s.replace('-AMP-', '&').replace('-LAB-', '<')
                         .replace('-RAB-', '>').replace(r'\*', '*')
                         .replace(r'[background_noise_', '')) # remove stupid mismatches
                    if (dep_content is not None and (curr_index >= len(dep_content[sent])
                            or dep_content[sent][curr_index]['token'] != s)):
                        if not re.match(r'^(\*([A-Z\?]+\*)?(-\d+)?|0)$', s):
                            print(sent, dep_content[sent][curr_index]['token'], s)
                            raise ValueError("content in .parse file and .coref file don't match")
                        # else ignore
                    else:
                        tokens.append(s)
                        curr_index += 1
#             print(sent, ' --> ', tokens)
#             print(sent, len(dep_content), line)
            assert dep_content is None or len(tokens) == len(dep_content[sent])
            sent += 1
    return chains

def load_named_entities(path, dep_content):
    names = defaultdict(list)
    sent = 0
    if not os.path.exists(path):
        print('Name data not found: %s' %path, file=sys.stderr)
    else:
        with codecs.open(path, 'r', 'utf-8') as f:
            for line in f:
                if line.strip() and not re.match(r'</?DOC|</?TEXT', line):
                    tokens = []
                    brackets = []
                    curr_index = 0
                    parts = re.findall(r'<ENAMEX\s[^>]+>|</ENAMEX\s*>|[^\s<>]+', line)
                    for s in parts:
                        m = re.match(r'<ENAMEX\s+TYPE="([^"]+)"', s)
                        if m:
                            brackets.append((m.group(1), curr_index))
                        elif re.match(r'</ENAMEX', s):
                            type_, start_index = brackets.pop()
                            for i in range(start_index, curr_index):
                                names[(sent, i)].append(type_)
                        else:
                            if curr_index >= len(dep_content[sent] or dep_content[sent][curr_index]['token'] != s):
                                if not re.match(r'^(\*([A-Z\?]+\*)?(-\d+)?|0)$', s):
                                    print(sent, dep_content[sent][curr_index]['token'], s)
                                    raise ValueError("content in .name file and .coref file don't match")
                                # else ignore
                            else:
                                tokens.append(s)
                                curr_index += 1
                    assert dep_content is None or len(tokens) == len(dep_content[sent])
                    sent += 1
    return names

class OntonotesDocument:
    
    def __init__(self, base_path):
        self.base_path = base_path
        self._trees = None
        self._deps = None
        self._named_entities = None
        self._frames = None
        self._chains = None
        self._coref = None
        self._supersenses = None

    def supersenses(self):
        if self._supersenses is None:
            self._supersenses = {}
            path = self.base_path + '.sense'
            if os.path.exists(path):
                with open(path) as f:
                    for line in f:
                        fields = line.strip().split(' ')
                        sent = int(fields[1])
                        token_index = int(fields[2])
                        word_pos = fields[3].split('-')
                        word, pos = '-'.join(word_pos[:-1]), word_pos[-1]
                        try:
                            sense_num = int(fields[5] if len(fields) > 5 else fields[4])
                            synset = wn.synset('%s.%s.%d' %(word, pos, sense_num))
                            supersense = synset.lexname()
                            self._supersenses[(sent, token_index)] = supersense
                        except (ValueError, WordNetError):
                            pass # probably added sense, ignore
            else:
                sys.stderr.write('File not found: %s\n' %path)
        return self._supersenses
    
    def supersense_of(self, sent, token_index, follow_coref=True, verbose=True, first_sense_only=True):
        pos = self.deps()[sent][token_index]['pos']
        supersense = self.supersenses().get((sent, token_index))
        if not supersense:
            try:
                token = self.deps()[sent][token_index]['token']
                synsets = wn.synsets(token)
                wn_pos = pos.replace('JJ', 'a')[0].lower()
                synsets = [s for s in synsets if s.pos().replace('s', 'a') == wn_pos]
                if len(synsets) >= 1:
                    if first_sense_only:
                        supersense = synsets[0].lexname()
                    else:
                        supersense = (s.lexname() for s in synsets)
            except UnicodeDecodeError:
                pass # nothing I can do
        if not supersense and follow_coref:
            chain = self.coref().get((sent, token_index)) or []
            for c in chain:
                supersense = self.supersense_of(*c, follow_coref=False, verbose=False)
                if supersense: break 
        if not supersense:
            if 'NN' in pos: # all nouns
                supersense = 'noun.Tops'
            elif 'JJ' in pos: # all adj 
                supersense = 'adj.all'
            elif 'RB' in pos:
                supersense = 'adv.all'
            elif 'PRP' in pos and token.lower() in ('i', 'me', 'myself', 'you', 
                                                    'he', 'his', 'him', 'himself', 
                                                    'she', 'her', 'herself',
                                                    'we', 'our', 'us', 'ourselves'):
                supersense = 'noun.person' 
        if first_sense_only:
            return supersense
        else:
            return set(supersense)

    def trees(self):
        if self._trees is None: 
            self._trees = ptb.load_trees(self.base_path + '.parse')
        return self._trees

    def deps(self):
        if self._deps is None:
            self._deps = conll.load_trees(self.base_path + '.dep')
        return self._deps

    def named_entities(self):
        if self._named_entities is None:
            self._named_entities = load_named_entities(self.base_path + '.name', self.deps())
        return self._named_entities

    def coref(self):
        if self._coref is None:
            path = self.base_path + '.coref'
            if os.path.exists(path):
                with codecs.open(path, 'r', 'utf-8') as f:
                    deps = self.deps()
                    self._chains_as_spans = load_coref(f.read(), deps)
                    self._chains = {} 
                    for refid in self._chains_as_spans:
                        heads = []
                        for sent, start, stop in self._chains_as_spans[refid]:
                            head = conll.find_head(range(start, stop), deps[sent])
                            if head is not None:
                                heads.append((sent, head))
                            else:
                                sys.stderr.write('<coref> head not found for "%s" in sentence %d, file %s\n'
                                                 %(' '.join(deps[sent][i]['token'] 
                                                            for i in range(start, stop)), sent, path))
                        self._chains[refid] = tuple(heads)
                    self._coref_as_spans = {}
                    for chain in self._chains_as_spans.values():
                        for h in chain:
                            self._coref_as_spans[h] = chain
                    self._coref = {}
                    for chain in self._chains.values():
                        for h in chain:
                            self._coref[h] = chain
            else:
                sys.stderr.write('File not found: %s, assume no coreference.\n' %path)
                self._chains_as_spans, self._coref_as_spans, self._chains, self._coref = {}, {}, {}, {}
        return self._coref

    @classmethod
    def _resolve_coordinate(cls, tree, terminal_number, height):
        u = tree.terminals[terminal_number]
        while height > 0: 
            u = u.parent()
            height -= 1
        return u
    
    @classmethod
    def _resolve_pointer(cls, tree, pointer):
        m14 = re.match(r'^\d+:\d+([,;\*]\d+:\d+)*\*?$', pointer)
        if m14: # form 1-4
            chain_parts = re.split('\*|;', pointer) # trace-chain operator has lower precedence
            coreferent_terminals = []
            for cp in chain_parts:
                split_parts = cp.split(',') 
                terminals = []
                for sp in split_parts:
                    m = re.match(r'^(\d+):(\d+)$', sp)
                    terminal_number = int(m.group(1))
                    height = int(m.group(2))
                    terminals.extend(cls._resolve_coordinate(tree, terminal_number, height).terminals)
                coreferent_terminals.append(terminals)
            return coreferent_terminals
        else:
            return None
        
    @property
    def frames(self):
        '''
        Return a list of @data.Frame objects.
        '''
        if self._frames is not None:
            return self._frames
        self._frames = []
        if (os.path.exists(self.base_path + '.prop')
                and os.path.exists(self.base_path + '.parse')
                and os.path.exists(self.base_path + '.dep')):
            print(self.base_path + '.parse')
            assert len(self.trees()) == len(self.deps()), \
                    'constituent and dependency trees mismatch (%s)' %self.base_path
            with open(self.base_path + '.prop') as inp:
                for line in inp:
                    fields = line.strip().split(' ')
                    sent = int(fields[1])
                    term = int(fields[2])
                    offset = self.trees()[sent].terminals[term].token_id
                    f = Frame(sent, offset, fields[5])
                    for role_str in fields[8:]:
                        if '-rel' not in role_str.lower() and '-support' not in role_str.lower():
                            fields = role_str.split('-')
                            pointer, rname = fields[0], '-'.join(fields[1:])
                            if 'link-' not in rname.lower(): # ignore links
                                coreferent_terminals = self._resolve_pointer(self.trees()[sent], pointer) 
                                if coreferent_terminals is not None:
                                    tokens = []
                                    heads = []
                                    strs = []
                                    for terminals in coreferent_terminals:
                                        token_ids = set(t.token_id for t in terminals if t.token_id is not None)
                                        if token_ids:
                                            tokens.append(token_ids)
                                            head = conll.find_head(token_ids, self.deps()[sent])
                                            if head is None:
                                                sys.stderr.write('head not found: %s\n'
                                                                 %" ".join(t[0] for t in terminals))  
                                            elif head != '__INGORE__':
                                                heads.append(head)
                                            strs.append(' '.join(self.deps()[sent][t]['token'] for t in token_ids))
                                    coref_indices = set()
                                    for terminals in coreferent_terminals:
                                        true_terminals = [t for t in terminals if t.token_id is not None]
                                        if any(true_terminals):
                                            start_idx = min(t.token_id for t in true_terminals)
                                            stop_idx = max(t.token_id for t in true_terminals)+1
                                            if stop_idx-start_idx == len(true_terminals): # contiguous 
                                                self.coref() # cause coref maps to be built
                                                chain = self._coref_as_spans.get((sent, start_idx, stop_idx))
                                                if chain:
                                                    coref_indices.update(chain)
                                                else:
                                                    coref_indices.add((sent, start_idx, stop_idx)) 
                                    coref_heads = []
                                    coref_strs = []
                                    for s, ss, se in coref_indices:
                                        men_str = ' '.join(d['token'] for d in self.deps()[s][ss:se])
                                        head = conll.find_head(range(ss, se), self.deps()[s])
                                        if head: coref_heads.append((s, head))
                                        coref_strs.append(men_str)
                                    f.frame_elements.append(FrameElement(rname, tokens, heads, strs, 
                                                                         coref_heads, coref_strs))
                                else:
                                    sys.stderr.write('Ignored pointer: %s\n' %pointer)
                    for fe in f.frame_elements:
                        fe.role = re.sub(r'(A\d+)-\w+', r'\1', # remove -PRD and the like after A0, A1,...  
                                          re.sub('-H\d+', '',  # remove hyphen tags
                                                 fe.role.replace('ARG', 'A')))
                    self._frames.append(f)        
        return self._frames
            
    def iter_frames(self):
        return iter(self.frames)

def list_docs(dir_path=ontonotes_en, shuffle=False, filter_func=None):
    paths = []
    for root, _, fnames in os.walk(dir_path):
        for fname in fnames:
            if fname.endswith('.parse'):
                paths.append(os.path.join(root, fname[:-6]))
    if shuffle:
        random.shuffle(paths)
    else:
        paths.sort() # remove indeterminism
    if filter_func is not None:
        paths = filter(filter_func, paths)
    return paths

