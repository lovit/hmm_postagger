from collections import defaultdict
import json
import re

doublespace_pattern = re.compile(u'\s+', re.UNICODE)

class TrainedHMMTagger:
    def __init__(self, model_path=None, transition=None, emission=None,
        begin=None, transition_smoothing=1e-8, emission_smoothing=1e-8,
        unknown_penalty=-10, end_state='EOS'):

        self.transition = transition if transition else {}
        self.emission = emission if emission else {}
        self.begin = begin if begin else {}
        self._ts = transition_smoothing
        self._es = emission_smoothing
        self.unknown_penalty = unknown_penalty
        self.end_state = end_state

        if isinstance(model_path, str):
            self.load_model_from_json(model_path)

        # for user-dictionary, max value
        self._max_score = {
            state:max(observations.values())
            for state, observations in self.emission.items()
        }

        self._max_word_len = 12 # default

    def load_model_from_json(self, model_path):
        with open(model_path, encoding='utf-8') as f:
            model = json.load(f)
        self.emission = model['emission']
        self.transition = model['transition']
        self.transition = {tuple(states.split()):prob for states, prob in self.transition.items()}
        self.begin = model['begin']
        del model

    def append_user_dictionary(self, tag, words):
        if not (tag in self.emission):
            raise ValueError('%s does not exist in tagset' % tag)
        append_score = self._max_score[tag]
        for word in words:
            self.emission[tag][word] = append_score

    def decode(self, sequence):
        raise NotImplemented

    def log_probability(self, sequence):
        # emission probability
        log_prob = sum(
            (self.emission.get(t, {}).get(w,self.unknown_penalty)
             for w, t in sequence)
        )

        # bos
        log_prob += self.begin.get(sent[0][1], self.unknown_penalty)

        # transition probability
        transitions = [(t0, t1) for (_, t0), (_, t1) in zip(sequence, sequence[1:])]
        log_prob += sum(
            (self.transition.get(transition, self.unknown_penalty)
             for transition in transitions))

        # eos
        log_prob += self.transition.get(
            (sequence[-1][1], self.end_state), self.unknown_penalty
        )

        # length normalization
        log_prob /= len(sequence)

        return log_prob

    def _get_pos(self, word):
        return [tag for tag, words in self.emission.items() if word in words]

    def _lookup(self, sentence):

        def word_lookup(eojeol, offset):
            n = len(eojeol)
            words = [[] for _ in range(n)]
            for b in range(n):
                for r in range(1, self._max_word_len+1):
                    e = b+r
                    if e > n:
                        continue
                    sub = eojeol[b:e]
                    for pos in self._get_pos(sub):
                        words[b].append((sub, pos, b+offset, e+offset))
            return words

        sentence = doublespace_pattern.sub(' ', sentence)
        sent = []
        for eojeol in sentence.split():
            sent += word_lookup(eojeol, offset=len(sent))
        return sent

    def _generate_edge(self, sentence):

        def get_nonempty_first(sent, end, offset=0):
            for i in range(offset, end):
                if sent[i]:
                    return i
            return offset

        chars = sentence.replace(' ','')
        sent = self._lookup(sentence)
        n_char = len(sent) + 1
        sent.append([('EOS', 'EOS', n_char, n_char)])

        nonempty_first = get_nonempty_first(sent, n_char)
        if nonempty_first > 0:
            sent[0].append((chars[:nonempty_first], 'Unk', 0, nonempty_first))

        graph = []
        for words in sent[:-1]:
            for word in words:
                begin = word[2]
                end = word[3]
                if not sent[end]:
                    b = get_nonempty_first(sent, n_char, end)
                    unk = (chars[end:b], 'Unk', end, b)
                    graph.append((word, unk))
                for adjacent in sent[end]:
                    graph.append((word, adjacent))

        unks = {node for _, node in graph if node[1] == 'Unk'}
        for unk in unks:
            for adjacent in sent[unk[3]]:
                graph.append((unk, adjacent))
        bos = ('BOS', 'BOS', 0, 0)
        for word in sent[0]:
            graph.append((bos, word))
        graph = sorted(graph, key=lambda x:(x[0][2], x[1][3]))

        return graph