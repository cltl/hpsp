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

dev_docs = set([
    'bc/p2.5_a2e/00/p2.5_a2e_0022','bc/p2.5_a2e/00/p2.5_a2e_0009','bc/p2.5_c2e/00/p2.5_c2e_0008','bc/p2.5_a2e/00/p2.5_a2e_0029','bc/p2.5_c2e/00/p2.5_c2e_0041',
    'bn/voa/00/voa_0016','bn/cnn/01/cnn_0173','bn/cnn/02/cnn_0272','bn/voa/00/voa_0043','bn/voa/01/voa_0144',
    'mz/sinorama/10/ectb_1037', 'mz/sinorama/10/ectb_1071', 'mz/sinorama/10/ectb_1051', 'mz/sinorama/10/ectb_1061', 'mz/sinorama/10/ectb_1057',
    'nw/wsj/11/wsj_1129', 'nw/wsj/05/wsj_0519', 'nw/wsj/15/wsj_1535', 'nw/wsj/10/wsj_1045', 'nw/wsj/07/wsj_0726',
    'pt/nt/40/nt_4012', 'pt/nt/53/nt_5303', 'pt/ot/10/ot_1022', 'pt/ot/11/ot_1110', 'pt/nt/58/nt_5806',
    'tc/ch/00/ch_0026', 'tc/ch/00/ch_0021', 'tc/ch/00/ch_0027', 'tc/ch/00/ch_0002', 'tc/ch/00/ch_0022',
    'wb/sel/72/sel_7260', 'wb/sel/78/sel_7818', 'wb/sel/09/sel_0942', 'wb/sel/52/sel_5296', 'wb/sel/58/sel_5861'
    ])


    
def load_coref(content, dep_content=None):
    chains = defaultdict(list)
    sent = 0
    for line in content.split('\n'):
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
                    chains[refid].append((sent, start_index, curr_index))
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

class OntonotesDocument:
    
    def __init__(self, base_path):
        self.base_path = base_path
        self._trees = None
        self._deps = None
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

def ontonotes2tiger(onto_doc):
    corpus = etree.Element("corpus")
    etree.SubElement(corpus, 'head')
    body = etree.SubElement(corpus, 'body')
    sent2sem = {}
    for tree_id, tree in enumerate(onto_doc.trees()):
        dep = onto_doc.deps()[tree_id]
        s_id = 's%d' %tree_id
        s = etree.SubElement(body, 's', id=s_id)
        # syntax trees
        graph = etree.SubElement(s, 'graph')
        terminals = etree.SubElement(graph, 'terminals')
        tree_terminals = [t for t in tree.terminals if t.token_id is not None]
        for i, tree_term in enumerate(tree_terminals):
            word = word=dep[i]['token']
            pos = tree_term.label()
            lemma = lemmatizer.lemmatize(word, get_wordnet_pos(pos) or 'n')
            t_id = '%s_%d' %(s_id, i)
            tree_term.nt_id = t_id
            etree.SubElement(terminals, 't', id=t_id, word=word, pos=pos, lemma=lemma)
        nonterminals = etree.SubElement(graph, 'nonterminals')
        
        def traverse_tree(i, st):
            if isinstance(st, str): return i
            for child in st:
                i = traverse_tree(i, child)
            token_ids = set(t.token_id for t in st.terminals 
                            if t.token_id is not None)
            if st.height() > 2 and token_ids:
                nt_id = '%s_%d' %(s_id, 500+i)
                nt = etree.SubElement(nonterminals, 'nt', id=nt_id, cat=st.label())
                head_id = find_syntactic_head(token_ids, dep)
                if head_id is None:
                    nt.attrib['head'] = '--'
                else:
                    nt.attrib['head'] = dep[head_id]['token']
                for child in st:
                    if hasattr(child, 'nt_id'):
                        etree.SubElement(nt, 'edge', idref=child.nt_id, label='-')
                    else:
                        pass
                st.nt_id = nt_id
                i += 1
            return i
        traverse_tree(0, tree)
                
        graph.attrib['root'] = tree.nt_id
        sem = etree.SubElement(s, 'sem')
        sent2sem[tree_id] = sem
        etree.SubElement(sem, 'globals')
    # semantic roles
    frame_elems = {}
    for onto_frame_id, onto_frame in enumerate(onto_doc.frames):
        assert onto_frame.sent >= 0 and onto_frame.offset >= 0
        tree = onto_doc.trees()[onto_frame.sent]
        sem = sent2sem[onto_frame.sent]
        frame_id = 's%d_f%d' %(onto_frame.sent, onto_frame_id)
        frame = etree.SubElement(sem, 'frame', name=onto_frame.predicate, id=frame_id)
        frame_elems[(onto_frame.sent, onto_frame.offset)] = frame
        target = etree.SubElement(frame, 'target')
        etree.SubElement(target, 'fenode', idref='s%d_%d' %(onto_frame.sent, onto_frame.offset))
        for onto_fe_id, onto_fe in enumerate(onto_frame.frame_elements):
            fe_id = '%s_e%d' %(frame_id, onto_fe_id)
            fe = etree.SubElement(frame, 'fe', id=fe_id, name=onto_fe.role)
            if onto_fe.filler_tokens:
                fe_nt = ptb.find_constituent(onto_fe.filler_tokens[0], tree)
                etree.SubElement(fe, 'fenode', idref=fe_nt.nt_id)
    # coreference
    coref_count = 0
    onto_doc.coref()
    for chain in onto_doc._chains_as_spans.values():
        coref_sent, coref_start, coref_end = chain[0]
        coref_nt = ptb.find_constituent(list(range(coref_start, coref_end)), 
                                        onto_doc.trees()[coref_sent])
        for onto_current in chain[1:]:
            curr_sent, curr_start, curr_end = onto_current
            curr_nt = ptb.find_constituent(list(range(curr_start, curr_end)), 
                                           onto_doc.trees()[curr_sent])
            if hasattr(curr_nt, 'nt_id') and hasattr(coref_nt, 'nt_id'):
                sem = sent2sem[curr_sent]
                
                frame_id = 's%d_c%d' %(curr_sent, coref_count)
                frame = etree.SubElement(sem, 'frame', name="Coreference", id=frame_id)
                target = etree.SubElement(frame, 'target')
                etree.SubElement(target, 'fenode', idref=curr_nt.nt_id)
                
                fe1 = etree.SubElement(frame, 'fe', name="Coreferent")
                etree.SubElement(fe1, 'fenode', idref=coref_nt.nt_id)
                
                fe2 = etree.SubElement(frame, 'fe', name="Current")
                etree.SubElement(fe2, 'fenode', idref=curr_nt.nt_id)
                coref_count += 1
    return corpus, frame_elems


if __name__ == '__main__':
    table = [['Predicate', 'Frame', 'Role', 'Filler', 'Filler_corefs']]
    for onto_doc in iter_docs(shuffle=True):
        xml_doc, _ = ontonotes2tiger(onto_doc)
        doc = Document(xml_doc)
        for frame_elem in doc.frames:
            if 'bring' in frame_elem.get('name'):
                for fe_elem in frame_elem.iterfind('fe'):
                    if fe_elem.find('fenode') is not None:
                        filler_id = fe_elem.find('fenode').get('idref')
                        filler_corefs = doc.coref.get(filler_id, [])
                        table.append([' '.join(doc.id2words[frame_elem.find('target/fenode').get('idref')]),
                                      frame_elem.get('name'),
                                      fe_elem.get('name'),
                                      ' '.join(doc.id2words[filler_id]),
                                      '='.join(' '.join(doc.id2words[fcoref]) 
                                                        for fcoref in filler_corefs)])
        if len(table) >= 1001:
            break;
    w = csv.writer(sys.stdout)
    w.writerows(table)    
