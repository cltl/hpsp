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

auxiliary_frames = set(['Coreference', 'Support', 'Relativization'])

class Stats:
    
    def __init__(self, name):
        self.name = name
        self.frame_num = 0
        self.noun_frame_num = 0
        self.verb_frame_num = 0
        self.dni_frame_num = 0
        self.dni_noun_frame_num = 0
        self.dni_verb_frame_num = 0
        self.dni_frame_in_ontonotes_num = 0
        self.dni_noun_frame_in_ontonotes_num = 0
        self.dni_verb_frame_in_ontonotes_num = 0
        self.frame_types = set()
        self.sentence_num = 0
        self.token_num = 0
        self.ini_num = 0
        self.dni_num = 0
        self.resolved_num = 0
        self.overt_num = 0
        self.dni_role_count = Counter()
        self.ontonotes_indexer = None
        p = 'output/ontonotes-frames.index'
        if os.path.exists(p):
            with open(p, 'rb') as f:
                self.ontonotes_indexer = pickle.load(f)

    def collect_data(self, in_path, reader):
        if not re.search(r"\.(xml|txt)$", in_path): return
        doc = Document(in_path, reader)
        self.sentence_num += sum(1 for _ in doc.elem.iterfind('.//s'))
        self.token_num += sum(1 for _ in doc.elem.iterfind('.//t'))
        for elem in doc.elem.iterfind('.//frame'):
            frame = elem.get('name')
            if elem.get('id') and frame not in auxiliary_frames:
                self.frame_num += 1
                tar_id = elem.find('target/fenode').get('idref')
                pos = doc.id2elem[tar_id].get('pos')
                if pos[0] == 'N': 
                    self.noun_frame_num += 1
                elif pos[0] == 'V':
                    self.verb_frame_num += 1
                if elem.find(".//flag[@name='Definite_Interpretation']") is not None:
                    self.dni_frame_num += 1
                    if pos[0] == 'N': 
                        self.dni_noun_frame_num += 1
                    elif pos[0] == 'V':
                        self.dni_verb_frame_num += 1
                    if self.ontonotes_indexer and frame in self.ontonotes_indexer.vocab:
                        self.dni_frame_in_ontonotes_num += 1
                        if pos[0] == 'N': 
                            self.dni_noun_frame_in_ontonotes_num += 1
                        elif pos[0] == 'V':
                            self.dni_verb_frame_in_ontonotes_num += 1
                    
                self.frame_types.add(elem.get('name'))
                for fe_elem in elem.findall('fe'):
                    if fe_elem.find("flag[@name='Indefinite_Interpretation']") is not None:
                        self.ini_num += 1
                    elif fe_elem.find("flag[@name='Definite_Interpretation']") is not None:
                        self.dni_num += 1
                        self.dni_role_count[fe_elem.get('name')] += 1
                        if fe_elem.find('fenode') is not None: 
                            self.resolved_num += 1
                    else:
                        self.overt_num += 1
            
    def run(self, dirs_or_files, reader=None):
        call_for_all_children(dirs_or_files, lambda path: self.collect_data(path, reader))
        print("*** Dataset: %s ***" %self.name)
        print("Sentences: %d" %self.sentence_num)
        print("Tokens: %d" %self.token_num)
        print("Frames: %d (noun: %d, verb: %d)" %(self.frame_num, self.noun_frame_num, self.verb_frame_num))
        print("Frames with DNI: %d (noun: %d, verb: %d)" %(self.dni_frame_num, self.dni_noun_frame_num, self.dni_verb_frame_num))
        if self.ontonotes_indexer:
            print(("... and found in Ontonotes: %d (noun: %d, verb: %d)" 
                   %(self.dni_frame_in_ontonotes_num, self.dni_noun_frame_in_ontonotes_num, self.dni_verb_frame_in_ontonotes_num)))
        print("Frame types: %d" %len(self.frame_types))
        print("Overt FEs: %d" %self.overt_num)
        print("DNIs (resolved): %d (%d)" %(self.dni_num, self.resolved_num))
        print("INIs: %d" %self.ini_num)
        print("Roles of DNIs:" + str(sorted(self.dni_role_count.items(), key=lambda x: -x[1])))


def print_overt_roles_this_span_play(doc, span_id):
    frames_roles = []
    for ovr_id in doc.id2ovrs[span_id]:
        ovr_fe_elem = doc.id2elem[ovr_id]
        ovr_frame_elem = ovr_fe_elem.getparent()
        frames_roles.append(ovr_frame_elem.get('name') + 
                            '_' + ovr_fe_elem.get('name'))
    if frames_roles:
        print(", ".join(frames_roles))
    else:
        print("(no role)")


def show_examples(path, num, reader=None):
    doc = Document(path, reader)
    for e in doc.elem.iterfind('.//frame'):
        if e.get('id') and e.get('name') not in auxiliary_frames:
            predicate = ' '.join(doc.id2words[e.find("target/fenode").get("idref")])
            for fe_elem in e.iterfind('fe'):
                if num == 0: return
                if (fe_elem.find("flag[@name='Definite_Interpretation']") is not None 
                        and fe_elem.find("fenode") is not None):
                    idref = fe_elem.find("fenode").get("idref")
                    print("************")
                    s = doc.id2sent[e.get('id')]
                    for i in range(max(0, s-2), s+1):
                        print("sentence %d: %s" %(i-s, " ".join(doc.id2words[doc.sentences[i].get('id')])))
                    print(e.get('id'), fe_elem.get('id'))
                    print("Frame: %s (%s)" %(e.get("name"), predicate))
                    for fe_elem2 in e.iterfind('fe'):
                        if (fe_elem2.find("flag[@name='Definite_Interpretation']") is None and
                            fe_elem2.find("flag[@name='Indefinite_Interpretation']") is None and
                            fe_elem2.find("fenode") is not None):
                            idref2 = fe_elem2.find("fenode").get("idref")
                            print(("Overt role of the same predicate: %s (%s, %s)" 
                                   %(fe_elem2.get("name"), doc.id2head[idref2], 
                                     ' '.join(doc.id2words[idref2]))))
                    print("FE: %s (%s, %s)" %(fe_elem.get("name"), doc.id2head[idref], 
                                              ' '.join(doc.id2words[idref])))
#                     sys.stdout.write("Overt roles this span play: ")  
#                     print_overt_roles_this_span_play(doc, idref)
                    if idref in doc.coref:
                        for coref_id in doc.coref[idref].difference({idref}):
                            target_sent = int(idref.split('_')[0][1:])
                            coref_sent = int(coref_id.split('_')[0][1:])
                            if (-2 <= coref_sent - target_sent <= 0 # previous two sentences
                                    and coref_id in doc.id2ovrs): 
                                sys.stdout.write("Coref: %s, %s" %(doc.id2head[coref_id], 
                                                                      ' '.join(doc.id2words[coref_id])))
                                print('')
#                                 print_overt_roles_this_span_play(doc, coref_id)
                    num -= 1

class Document:
    '''
    This class wraps around a Tiger XML document and provide additional data
    structure for easy access to information.
    '''
    
    def __init__(self, xml_doc, reader=None):
        if isinstance(xml_doc, str):
            path = xml_doc
            reader = reader or etree.parse
            self.elem = reader(path)
        else:
            path = None
            self.elem = xml_doc
        self.sentences = self.elem.findall(".//s")
        self.frames = self.elem.findall(".//frame")
        self.id2elem = {} # for fast access
        self.id2sent = {}
        for i, s in enumerate(self.elem.iterfind(".//s")):
            self.id2elem[s.get("id")] = s
            for e in s.iterfind('.//*[@id]'):
                self.id2elem[e.get("id")] = e
                self.id2sent[e.get("id")] = i
        self.id2words = {} # recursively search for words will be crazily slow
        for sent in self.sentences:
            words = []
            for t in sent.iterfind(".//t"):
                words.append(t.get('word'))
            self.id2words[sent.get('id')] = words
        self.id2head = {} # id to head (a terminal node) id
        for e in self.elem.iterfind('.//part'):
            self.id2words[e.get("id")] = [unescape(e.get("word"))]
        for e in self.elem.iterfind('.//t'):
            self.id2words[e.get("id")] = [unescape(e.get("word"))]
            self.id2head[e.get("id")] = e.get("id")
        self.id2sparent = {}
        self.id2schildren = defaultdict(list)
        for e in self.elem.iterfind('.//nt'):
            words = []
            for edge in sorted(e.iterfind("edge"), key=lambda e: e.get('idref')):
                words += self.id2words[edge.get("idref")]
                self.id2sparent[edge.get("idref")] = e.get('id')
                self.id2schildren[e.get('id')].append(edge.get("idref"))
            self.id2words[e.get("id")] = words
            head_elem = self._search_head(e, e.get("head"))
            if not (head_elem is not None or e.get('head') == '--'):
                print('!!!')
            assert head_elem is not None or e.get('head') == '--'
            if head_elem is not None:
                self.id2head[e.get("id")] = head_elem.get("id")
        
        self.coref = {}
        for e in self.elem.findall('.//frame[@name="Coreference"]'):
            # collect all possible refs
            coref_refs = {e.find('target/fenode').get("idref")}
            current_elem = e.find('fe[@name="Current"]/fenode')
            if current_elem is not None:
                coref_refs.add(current_elem.get("idref"))
            coref_elem = e.find('fe[@name="Coreferent"]/fenode')
            if coref_elem is not None:
                coref_refs.add(coref_elem.get("idref"))
            if current_elem is None and coref_elem is None:
                sys.stderr.write("Missing attribute in a fenode of frame %s, skipped.\n" %e.get('id'))
            # add other coreferring refs 
            coref_set = set(coref_refs)
            for ref in coref_refs:
                if ref in self.coref:
                    coref_set.update(self.coref[ref])
            # update map to point to the new set 
            for ref in coref_set:
                self.coref[ref] = coref_set
        self.id2ovrs = defaultdict(set) # constituent id --> overt roles
        for e in self.elem.iterfind('.//fe'):
            fenode = e.find('fenode')
            if (fenode is not None and e.getparent().get('name') != 'Coreference' 
                    and e.find('flag[@name="Definite_Interpretation"]') is None): 
                self.id2ovrs[fenode.get('idref')].add(e.get('id')) 
        self.id2supersense = {}
        for id_, elem in self.id2elem.items():
            if elem.get('pos') == 'NNP':
                self.id2supersense[id_] = 'noun.Tops'
        if path is not None:
            supersense_path = re.sub(r'\.txt$', '.supersense', path)
            if supersense_path != path and os.path.isfile(supersense_path):
                with open(supersense_path) as f:
                    for line in f:
                        fields = re.split(r' +', line)
                        self.id2supersense[fields[1]] = fields[2]
                    
    def _search_head(self, nt, head_word):
        for edge in nt.iterfind("edge"):
            child = self.id2elem[edge.get("idref")]
            assert child is not None
            if child.tag == 'nt':
                head_elem = self._search_head(child, head_word)
                if head_elem is not None: return head_elem
            elif child.tag == 't':
                if unescape(child.get('word')) == unescape(head_word):
                    return child
            else:
                raise "Illegal constituent: %s (id=%s)" %(child.tag, child.get("id"))
    
    def head_word(self, elem):
        if isinstance(elem, str):
            elem = self.id2elem[elem]
        w = elem.get('head') or elem.get('word')
        assert w is not None
        return w

    def nonpronoun_coref(self, ref):
        refid = ref.get("id") if isinstance(ref, etree.ElementBase) else ref 
        chain = self.coref.get(refid)
        if chain:
            for c in chain:
                if c in self.id2head: 
                    head_elem = self.id2elem[self.id2head[c]]
                    if head_elem.get('pos') != 'PRP':
                        return self.head_word(head_elem)
        return self.head_word(refid)
    
    def collect_tokens(self, idref):
        elem = self.id2elem[idref]
        if elem.tag == 't':
            return set([idref])
        else:
            tokens = set()
            for edge_elem in elem.iterfind('.//edge'):
                tokens.update(self.collect_tokens(edge_elem.get('idref')))
            return tokens

class PBArgmap:
    '''
    Ported from class PBArgmap in semeval10-v7
    '''
    
    def __init__(self, elem, prefix):
        self.roleset = elem.get(prefix + "roleset")
        self.h_role = {}
        for e in elem.iterfind("role"):
            self.h_role[e.get("fn-role")] = e.get(prefix + "role")


class Pbparser:
    '''
    Adopted from class propbank.pbparser.Pbparser in sem10scorer.
    '''

    def __call__(self, path):     
        self.corpus = etree.Element("corpus")
        self.head = etree.Element("head")
        self.body = etree.Element("body")
        self.meta = etree.Element("meta")
        self.author = etree.Element("author")
        self.cid = etree.Element("corpus_id")
        self.date = etree.Element("date")
        self.desc = etree.Element("description")
        self.format = etree.Element("format")
        self.hist = etree.Element("history")
        self.name = etree.Element("name")
        self.all_terminals = {}
        # coref-frames occur only in the gold standard, so we count coref IDs separately
        # coref IDs stadrt at 500 to clearly separate them from normal frames
        self.corefCount = 500
        # map strings of sentence-IDs to actual sentences
        self.sentenceMap = {}
        # map strings of frame-IDs to actual frames
        self.frameMap = {}
        self._parseCorpus(path)
        return self.corpus

    def _parseCorpus(self, aFile):
        '''
        parses a corpus from a given file
        '''
        print("Parsing PB-Corpus from " + aFile)
        self.corpus.set("name", aFile)
        self.corpus.set("target", "None")

        # needed to make the Corpus a valid FrameNet Corpus (those elements are
        # not needed in fact, but having them may avoid errors later on)
        self.author.text = "Automatically created from " + aFile
        self.cid.text = aFile
        self.date.text = datetime.today().strftime("%d.%m.%Y")
        self.desc.text = "Autmatic conversion from PB to Negra Format"
        self.format.text = "Negra Format, Version 4"
        self.name.text = "Version 1"
        self.meta.append(self.author)
        self.meta.append(self.cid)
        self.meta.append(self.date)
        self.meta.append(self.desc)
        self.meta.append(self.format)
        self.meta.append(self.hist)
        self.meta.append(self.name)
        self.head.append(self.meta)
#         self.head.append(new Frames())
#         self.head.append(new Wordtags())
        self.corpus.append(self.head)
        self.corpus.append(self.body)

        # initialize everything
#         corpusProcessor = new CorpusProcessor(corpus)
        
        # read the file
        with open(aFile) as reader:
            print("Parsing Sentences")
            # multiple lines for a single sentence, seperated by tabs
            lines = []
            for curLine in reader:
                # newlines seperate sentences
                curLine = curLine.strip()
                if curLine == '':
                    if len(lines) > 0:
                        # a new complete sentence is in lines, add it to the corpus
                        self.addToCorpus(lines)
                    del lines[:]
                else:
                    # split the current line and add the array to lines
                    lines.append(curLine.split("\t"))
    
        # read the file a second time, this time collecting the frames
        with open(aFile) as reader:
            # only needed for outpus
            foo = 0
            print("Parsing Frames")
            for curLine in reader:
                curLine = curLine.strip()
                if curLine != '':
                    # a new line possibly containing frames is found, clear the
                    # frame map and generate frames from that line
                    self.frameMap.clear()
                    if int(curLine.split("\t")[0]) != foo:
                        foo = int(curLine.split("\t")[0])
#                         if (foo % 30 == 0): print("")
#                         print("s" + str(foo) + ", ")
                    # frames can be found in rows 8 and 9
                    self.generateFrames(curLine.split("\t"), 7)
                    self.generateFrames(curLine.split("\t"), 8)
#                 /* else {
#                     frameCount = 0
#                     corefCount = 500
#                 */
#             print("")
            reader.close()
        return self.corpus

    def generateFrames(self, line, position):
        '''
        extract the frames of a given line
        '''
        # true if the row contains at least one frame
        if line[position] != "_":
            # get several IDs
            sentenceID = "s" + line[0]
            terminalID = sentenceID + "_" + line[1]

            # ";" separates frames as well as frame elements, so "|" is added
            # as a unique identifier for frames
            frameline = line[position].replace("};", "}|")

            # get the frames already generated for the current sentence
            curFrames = self.sentenceMap[sentenceID].find(".//frames")

            # iterate over all the frames of the current line
            for frameString in frameline.split("|"):
                # get the name of the frame
                frameName = frameString.split("{")[0]
                # content here means the set of IDs of Terminals the frame
                # contains
                # a frame may only be evoked, but not contain any Terminals
                if len(frameString.split("{")[1].split("}")) > 0:
                    frameContent = frameString.split("{")[1].split("}")[0]

                # see if we are adding new FrameElements to a Frame we have
                # already found
                # if not, create a new Frame
                if frameName not in self.frameMap:
                    # coref-Frames and normal Frames are assigned different IDs
                    if frameName.split(".")[0].lower() in ("coref", "coreference"):
                        self.frameMap[frameName] = etree.Element(
                                "frame", attrib={"name": "Coreference", "id": sentenceID + "_" + frameName})
                        # corefCount++
                    else:
                        self.frameMap[frameName] = etree.Element(
                                "frame", attrib={"name": frameName, "id": sentenceID + "_" + frameName})
                        # frameCount++
                    # the target of the frame is the Terminal that has evoked
                    # it
                    target = etree.SubElement(self.frameMap[frameName], 'target')
                    etree.SubElement(target, "fenode", attrib={"idref": terminalID})
                    # add the new frame to the list of frames of the current
                    # sentence
                    curFrames.append(self.frameMap.get(frameName))

                # get the Frame mapped to the frameName
                frame = self.frameMap[frameName]

                # add FrameElements to the Frame
                if frameContent:
                    # FEs are separated by ";"
                    for feString in frameContent.split(";"):
                        # get the name and the ID of the FrameElement
                        feName = feString.split("_")[0]
                        # String feID = feName.split("A")[1]

                        # coref-Elements get a distinct name (current or
                        # coreferent)
                        if frame.get('name') == "Coreference":
                            if feName == "A0": feName = "Current"
                            if feName == "A1": feName = "Coreferent"

                        # the FrameElement currently investigated may only
                        # extend an FE already created earlier
                        found = False
                        for fe in frame.findall("fe"):
                            # try to get a corresponding FE
                            if fe.get('name').lower() == feName:
                                found = True
                                # get the terminals spanned by the found FE
                                terminals = [terminal for terminal in 
                                             feString.split("(")[1].split(")")[0].split(",")]
                                # get the Nonterminal spanning all these
                                # terminals (if any)
                                ntID = self.getMaxNT(terminals)

                                # either add the Nonterminal to the FE or all
                                # the terminals, if no Nonterminal could be
                                # found
                                if ntID is not None:
                                    etree.SubElement(fe, "fenode", attrib={"idref": ntID})
                                else:
                                    for terminal in terminals:
                                        etree.SubElement(fe, "fenode", attrib={"idref": terminal})
                                break
                        # if we found a new FrameElement
                        if not found:
                            # create a new FE
                            ele = etree.SubElement(frame, 'fe', attrib={'id': frame.get('id') + "_" + feName,
                                                                        'name': feName})
                            # set some Flags
                            interpretation = feString.split("_")[1].split("=")[0]
                            if interpretation == "DNI":
                                etree.SubElement(ele, 'flag', attrib={'name': "Definite_Interpretation"})
                            elif interpretation == "INI":
                                etree.SubElement(ele, 'flag', attrib={'name': "Indefinite_Interpretation"})
                            if position == 8:
                                etree.SubElement(ele, 'flag', attrib={'name': "Constructional_licensor"})

                            # same procedure as above, get Terminals,
                            # corresponding NT and add them/it to the FE
                            terminals = [terminal for terminal in 
                                         feString.split("(")[1].split(")")[0].split(",")]
                            ntID = self.getMaxNT(terminals)
                            if ntID is not None:
                                etree.SubElement(ele, "fenode", attrib={'idref': ntID})
                            else:
                                for terminal in terminals:
                                    etree.SubElement(ele, "fenode", attrib={"idref": terminal})

    def addToCorpus(self, lines):
        '''
        adds the terminals and nonterminals of a given sentence to the corpus
        '''
        # generate a new sentence, its ID can be found in the first row
        curSentence = etree.Element("s")
        sentenceID = "s" + lines[0][0]  # the sentence ID is needed several times
        curSentence.set("id", sentenceID)
        # each sentence has a Graph, Terminals, Nonterminals, Semantics and Frames
        curGraph = etree.Element("graph")
        curTerminals = etree.Element("terminals")
        curNonterminals = etree.Element("nonterminals")
        curSem = etree.Element("sem")
        curFrames = etree.Element("frames")

        curSem.append(curFrames)
        curGraph.append(curTerminals)
        curGraph.append(curNonterminals)
        curSentence.append(curGraph)
        curSentence.append(curSem)

        # a sentence belongs to a corpus' body
        self.body.append(curSentence)

        # map the current sentence to its ID
        self.sentenceMap[sentenceID] = curSentence
        # Nonterminals start at count 500, to easily distinguish them from
        # Terminals (which start at 1)
        ntCounter = 500

        # stack for building Nonterminals
        ntStack = []

        # iterate over the Terminals of the current sentence
        for line in lines:
            # the Terminal's ID can be found in row 2
            terminalID = sentenceID + "_" + line[1]

            # generate a new Terminal and add it to the sentence's terminals
            etree.SubElement(curTerminals, "t", attrib={"id": terminalID,
                                                        "lemma": line[3],
                                                        "pos": line[4],
                                                        "word": line[2]})
            # row 7 of each line defines the syntax of the sentence
            syntax = line[6]

            # parsing the sentence's syntax. Each Terminal either opens,
            # belongs to or closes a Nonterminal
            # "(" opens a new Nonterminal
            if "(" in syntax:
                # this is for sentences that have not been parsed correctly and
                # only consist of (NONE) as structure
                if syntax == "(NONE":
                    newNT = etree.Element("nt",
                                             attrib={"cat": "NONE", 
                                                     "head": "--",
                                                     "id": sentenceID + "_" + str(ntCounter)})
                    ntStack.append(newNT)
                    ntCounter += 1
                # this is for sentences with a complete syntactic structure
                else:
                    for nt in syntax.split("("):
                        if nt != '':
                            # a Nonterminal has a category and a head
                            cat = nt.split(":")[0]
                            # either the current Terminal is the head
                            # (catch-case) or it is defined right behind the
                            # category
                            try:
                                head = lines[int(nt.split(":")[1]) - 1][2]
                            except IndexError:
                                head = line[2]
                            # create and add a new Nonterminal to the stack,
                            # increment the ntCounter
                            newNT = etree.Element("nt",
                                                  attrib={"cat": cat,
                                                          "head": head,
                                                          "id": sentenceID + "_" + str(ntCounter)})
                            ntStack.append(newNT)
                            ntCounter += 1
                # the Nonterminal created contains an edge to the current
                # Terminal
                etree.SubElement(ntStack[-1], "edge", attrib={"idref": terminalID,
                                                              "label": "-"})
            # a "*" signals a Terminal belonging to the current Nonterminal
            if "*" in syntax:
                etree.SubElement(ntStack[-1], "edge", attrib={"idref": terminalID,
                                                              "label": "-"})
            # closing Nonterminals
            while ")" in syntax:
                # add hierarchically dominated Nonterminals to the Nonterminals
                # dominating them
                if len(ntStack) >= 2:
                    etree.SubElement(ntStack[-2], "edge",
                                     attrib={"idref": ntStack[-1].get("id"),
                                             "label": "-"})
                else:
                    curGraph.set("root", ntStack[0].get("id"))
                l = []
                for edge in ntStack[-1].iterfind('edge'):
                    idref = edge.get('idref')
                    if idref in self.all_terminals:
                        l += self.all_terminals[idref]
                    else:
                        l.append(idref)
                self.all_terminals[ntStack[-1].get('id')] = l
                curNonterminals.append(ntStack[-1])    
                del ntStack[-1]
                syntax = syntax.replace(")", "", 1)
                

    def getMaxNT(self, terminals):
        '''
        returns the ID of the Nonterminal that spans a given list of Terminals
        '''
        # only Nonterminals of the sentence containing the terminals can span
        # the terminals
        sentence = self.sentenceMap[terminals[0].split("_")[0]]

        # for each Nonterminal, get the Terminals it spans and compare to the
        # given ArrayList
        for nt in sentence.find(".//nonterminals"):
            if self.all_terminals[nt.get('id')] == terminals:
                # return the Nonterminal's ID if the lists match
                return nt.get("id")

class Frame(object):
    
    def __init__(self, sent, offset, predicate, doc=''):
        self.doc = doc
        self.sent = sent
        self.offset = offset
        self.predicate = predicate
        self.frame_elements = []

class FrameElement(object):
    
    def __init__(self, role, filler_tokens, filler_heads, filler_strs, 
                 filler_coref_heads, filler_coref_strs, sent=-1):
        self.role = role
        self.filler_tokens = filler_tokens
        self.filler_heads = filler_heads
        self.filler_strs = filler_strs
        self.filler_coref_heads = filler_coref_heads 
        self.filler_coref_strs = filler_coref_strs
        self.sent = sent


def is_dni(fe_elem):
    return fe_elem.find("flag[@name='Definite_Interpretation']") is not None
                 
def is_ini(fe_elem):
    return fe_elem.find("flag[@name='Indefinite_Interpretation']") is not None

def is_ni(fe_elem):
    return is_dni(fe_elem) or is_ini(fe_elem)


def _find_all_descendent_ids(doc, root, list_ids=None):
    if list_ids is None: list_ids = []
#     list_ids = list_ids or []
    if root.tag == 't':
        list_ids.append(root.get('id'))
    elif root.tag == 'nt':
        for edge_elem in root.iterfind('edge'):
            child = doc.id2elem[edge_elem.get('idref')]
            _find_all_descendent_ids(doc, child, list_ids)
    else:
        raise "Unknown type: " + root.tag
    return list_ids

def add_fe_to_file(doc, fe_elems, in_path, out_path):
    loc2fes = defaultdict(list)
    for fe_elem in fe_elems:
        frame_elem = fe_elem.getparent()
        frame = frame_elem.get('name')
        idref = frame_elem.find('target/fenode').get('idref')
        m = re.match('s(\d+)_(\d+)', idref)
        sent_id = int(m.group(1))
        word_id = int(m.group(2))
        role = fe_elem.get('name')
        
        ids = ''
        arg_sent_id = -1
        if is_dni(fe_elem):
            interpretation = 'DNI'
        elif is_ini(fe_elem):
            interpretation = 'INI'
        else:
            raise ValueError('Fe XML element without interpretation flag.')
        # oddly enough, INIs do have a filler and the scorer will break
        # if you don't provide them 
        if fe_elem.find('fenode') is not None:
            arg_elem = doc.id2elem[fe_elem.find('fenode').get('idref')]
            ids = ",".join(_find_all_descendent_ids(doc, arg_elem))
            arg_sent_id = int(ids[1:ids.index('_')])
            if is_ini(fe_elem): print('Hey!', ids)
        fe_str = "%s_%s=(%s)" %(role, interpretation, ids)
        loc2fes[(sent_id, word_id, arg_sent_id == sent_id, frame)].append(fe_str)
    num_added_fe = 0
    with open(in_path, 'rt') as inp:    
        with open(out_path, 'wt') as out:
            for line in inp:
                if line.strip():
                    fields = line.strip().split('\t')
                    sent_id = int(fields[0])
                    word_id = int(fields[1])
                    for local in [True, False]:
                        frame_contents = fields[7 if local else 8]
                        if frame_contents != '_':
                            assert frame_contents[-1] == '}'
                            frame_contents = frame_contents[:-1].split('};')
                            for i in range(len(frame_contents)):
                                pos = frame_contents[i].index('{')
                                frame = frame_contents[i][:pos]
                                fes = frame_contents[i][pos+1:]
                                fes = fes.split(';') if fes else []
                                fes_to_add = loc2fes[(sent_id, word_id, local, frame)]
                                frame_contents[i] = frame + "{" + ";".join(sorted(fes + fes_to_add))
                                del loc2fes[(sent_id, word_id, local, frame)]
                                num_added_fe += len(fes_to_add)
                            frame_contents = '};'.join(frame_contents) + '}'
                            fields[7 if local else 8] = frame_contents
                    line = "\t".join(fields)
                out.write(line)
                out.write('\n')
            if num_added_fe != len(fe_elems):
                assert False, "Some FEs weren't added: " + str(loc2fes)

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
    
if __name__ == '__main__':
#     Stats('train').run(train_semeval_fn_path)
#     Stats('test_gold').run(gold_fn_path)
#     Stats('train').run(train_semeval_pb_path, Pbparser())
#     Stats('test_gold').run(gold_pb_path, Pbparser())
#     show_examples(train_semeval_fn_path, 50)
    show_examples("data/SEMEVAL/Semeval2010Task10TrainingFN+PB/Semeval2010Task10TrainingPB/correctedTiger.PB.stripped.txt", -1, Pbparser())
#     with open('/tmp/test.txt', 'wb') as f:
#         f.write(etree.tostring(pb_doc, pretty_print=True))
