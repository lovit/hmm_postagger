from collections import defaultdict
import json
import re

from .path import ford_list
from .lemmatizer import lemma_candidate
from .utils import bos as bos_state
from .utils import eos as eos_state
from .utils import unk as unk_state

doublespace_pattern = re.compile(u'\s+', re.UNICODE)

class TrainedHMMTagger:
    def __init__(self, model_path=None, transition=None,
        emission=None, acceptable_transition=None, no_inference_tags=None):

        self.transition = transition if transition else {}
        self.emission = emission if emission else {}
        self._max_word_len = 10 # default
        self._max_eomi_len = 3
        self._max_modifier_len = 4

        if no_inference_tags is None:
            no_inference_tags = [bos_state, eos_state, 'Pronoun', 'Number']
        self._no_inference_tags = no_inference_tags

        # 'eg) 좋은 노래야 -> [좋, 은, 노랗, 애야] vs [좋, 은, 노래, 야]'
        self._noun_preference = 3.0
        self._a_syllable_noun_penalty = 5.0

        if isinstance(model_path, str):
            self.load_model_from_json(model_path)
            self._initialize(acceptable_transition)
        elif (transition is not None) and (emission is not None):
            self._initialize(acceptable_transition)
        else:
            raise ValueError('Insert model path or transition and emission manually')

    def load_model_from_json(self, model_path):
        with open(model_path, encoding='utf-8') as f:
            model = json.load(f)
        self.emission = model['emission']
        self.transition = model['transition']
        self.transition = {tuple(states.split()):prob for states, prob in self.transition.items()}
        del model

    def _initialize(self, acceptable_transition):
        if not acceptable_transition and self.transition:
            # use trained transition
            acceptable_transition = set(self.transition.keys())
            # add unknown state transiton
            for prev_, next_ in self.transition.keys():
                acceptable_transition.add((prev_, unk_state))
                acceptable_transition.add((unk_state, next_))

        self.acceptable_transition = acceptable_transition
        self.unknown_word = min(
            min(words.values())/2 for words in self.emission.values())
        self.unknown_transition = min(self.transition.values())

        # for user-dictionary, max value
        self._max_score = {
            state:max(observations.values())
            for state, observations in self.emission.items()
        }

    def tag(self, sentence, inference_unknown=True):
        # lookup
        chars = sentence.replace(' ','')
        lookupeds = self._sentence_lookup(sentence)

        # generate candidates
        edges, bos, eos = self._generate_edge(chars, lookupeds)
        edges = self._add_weight(edges)
        nodes = {node for edge in edges for node in edge[:2]}

        # choose optimal sequence
        path, cost = ford_list(edges, nodes, bos, eos)

        pos = self._separate_morphemes(path)

        if inference_unknown:
            pos = self._inference_unknown(pos)

        pos = self._postprocess(pos)
        return pos

    def _separate_morphemes(self, path):
        pos = []
        for word, tag0, tag1, b, e in path:
            morphs = word.split(' + ')
            pos.append((morphs[0], tag0))
            if len(morphs) == 2:
                pos.append((morphs[1], tag1))
        return pos

    def _inference_unknown(self, pos):
        pos_ = []
        for i, pos_i in enumerate(pos[:-1]):
            # skip bos or known state
            if i == 0 or not (pos_i[1] == unk_state):
                pos_.append(pos_i)
                continue

            # previous -> current transition
            tag_prob = {
                tag:prob for (prev_tag, tag), prob in self.transition.items()
                if prev_tag == pos[i-1][1]
            }

            # current -> next transition
            for (tag, next_tag), prob in self.transition.items():
                if next_tag == pos[i+1][1]:
                    tag_prob[tag] = tag_prob.get(tag, 0) + prob

            for tag in self._no_inference_tags:
                if tag in tag_prob:
                    tag_prob.pop(tag)

            if not tag_prob:
                infered_tag = 'Noun'
            else:
                infered_tag = sorted(tag_prob, key=lambda x:-tag_prob[x])[0]

            pos_.append((pos_i[0], infered_tag))
        pos_.append(pos[-1])

        return pos_

    def _postprocess(self, pos):
        return pos[1:-1]

    def log_probability(self, sequence):
        # emission probability
        log_prob = sum(
            (self.emission.get(t, {}).get(w,self.unknown_word)
             for w, t in sequence)
        )

        # bos
        log_prob += self.transition.get(
            (bos_state, sent[0][1]), self.unknown_transition
        )

        # transition probability
        transitions = [(t0, t1) for (_, t0), (_, t1) in zip(sequence, sequence[1:])]
        log_prob += sum(
            (self.transition.get(transition, self.unknown_transition)
             for transition in transitions))

        # eos
        log_prob += self.transition.get(
            (sequence[-1][1], eos_state), self.unknown_transition
        )

        # length normalization
        log_prob /= len(sequence)

        return log_prob

    def _get_pos(self, word):
        return [tag for tag, words in self.emission.items() if word in words]

    def _sentence_lookup(self, sentence):
        sentence = doublespace_pattern.sub(' ', sentence)
        sent = []
        for eojeol in sentence.split():
            sent += self._word_lookup(eojeol, offset=len(sent))
        return sent

    def _word_lookup(self, eojeol, offset):
        n = len(eojeol)
        pos = [[] for _ in range(n)]
        for b in range(n):
            for r in range(1, self._max_word_len+1):
                e = b+r
                if e > n:
                    continue
                sub = eojeol[b:e]
                for tag in self._get_pos(sub):
                    pos[b].append((sub, tag, tag, b+offset, e+offset))
                for i in range(1, self._max_modifier_len + 1):
                    len_r = r - i
                    if not (0 <= len_r <= 2):
                        continue
                    try:
                        lemmas = self._lemmatize(sub, i)
                        if lemmas:
                            for sub_, tag0, tag1 in lemmas:
                                pos[b].append((sub_, tag0, tag1, b+offset, e+offset))
                    except:
                        continue
        return pos

    def _lemmatize(self, word, i):
        l = word[:i]
        r = word[i:]
        lemmas = []
        len_word = len(word)
        for l_, r_ in lemma_candidate(l, r):
            word_ = l_ + ' + ' + r_
            if (l_ in self.emission['Verb']) and (r_ in self.emission['Eomi']):
                lemmas.append((word_, 'Verb', 'Eomi'))
            if (l_ in self.emission['Adjective']) and (r_ in self.emission['Eomi']):
                lemmas.append((word_, 'Adjective', 'Eomi'))
            if len_word > 1 and not (word in self.emission['Noun']):
                if (l_ in self.emission['Noun']) and (r_ in self.emission['Josa']):
                    lemmas.append((word_, 'Noun', 'Josa'))
        return lemmas

    def _generate_edge(self, chars, sent):

        def get_nonempty_first(sent, end, offset=0):
            for i in range(offset, end):
                if sent[i]:
                    return i
            return offset

        n_char = len(sent)
        eos_end_index = n_char + 1
        eos = (eos_state, eos_state, eos_state, n_char, eos_end_index)
        sent.append([eos])

        # check first word is unknown
        nonempty_first = get_nonempty_first(sent, eos_end_index)
        if nonempty_first > 0:
            e = nonempty_first
            first_node = (chars[:e], unk_state, unk_state, 0, e)
            sent[0].append(first_node)

        last_end_index = 0
        edges = []
        for words in sent[:-1]:
            for word in words:
                begin = word[3]
                end = word[4]
                # if exist adjacent node
                if not sent[end]:
                    b = get_nonempty_first(sent, eos_end_index, end)
                    unk = (chars[end:b], unk_state, unk_state, end, b)
                    edges.append((word, unk))
                # else
                for adjacent in sent[end]:
                    if (word[1], adjacent[1]) in self.acceptable_transition:
                        edges.append((word, adjacent))
                # update last end index
                if last_end_index < end:
                    last_end_index = end

        # if last word is unknown
        if last_end_index < n_char:
            end = last_end_index
            unk_word = chars[end:n_char]
            unk = (unk_word, unk_state, unk_state, end, n_char)
            edges.append((unk, eos))

        # edge from unk to next node
        unks = {to_node for _, to_node in edges if to_node[1] == unk_state}
        for unk in unks:
            for adjacent in sent[unk[3]]:
                edges.append((unk, adjacent))

        # edge from bos to first node
        bos = (bos_state, bos_state, bos_state, 0, 0)
        for word in sent[0]:
            edges.append((bos, word))

        # sort by (begin index, end index)
        edges = sorted(edges, key=lambda x:(x[0][3], x[1][4]))

        return edges, bos, eos

    def _add_weight(self, edges):

        def weight(from_, to_):
            morphs = to_[0].split(' + ')

            # score of first word
            w = self.emission.get(to_[1], {}).get(morphs[0], self.unknown_word)
            if to_[1] == 'Noun': ## noun preference
                w /= self._noun_preference # because score is - log prob
                if len(morphs[0]) == 1:
                    w *= self._a_syllable_noun_penalty
            w += self.transition.get((from_[2], to_[1]), self.unknown_transition)

            # score of second word
            if len(morphs) == 2:
                w += self.emission.get(to_[2], {}).get(morphs[1], self.unknown_word)
                w += self.transition.get(to_[1], {}).get(to_[2], self.unknown_transition)
            return w

        graph = [(from_, to_, weight(from_, to_)) for from_, to_ in edges]
        return graph

    def add_user_dictionary(self, tag, words):
        if isinstance(words, str):
            words = [words]
        if not (tag in self.emission):
            raise ValueError('{} tag does not exist in model'.format(tag))
        append_score = self._max_score[tag]
        for word in words:
            self.emission[tag][word] = append_score