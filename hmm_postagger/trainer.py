from collections import defaultdict
import json
import math

from .utils import check_dirs

class CorpusTrainer:
    def __init__(self, tagset=None, min_count_tag=5, min_count_word=1, verbose=True):
        self.tagset = tagset
        self.min_count_tag = min_count_tag
        self.min_count_word = min_count_word
        self.verbose = verbose

    def train(self, corpus, model_path=None):

        pos2words, transition, bos = self._count_pos_words(corpus)

        self.pos2words_, self.transition_, self.bos_ = self._to_log_prob(
            pos2words, transition, bos)

        if model_path:
            if model_path[-4:] != 'json':
                model_path += '.json'
            self._save_as_json(model_path)

    def _count_pos_words(self, corpus):

        def trim_words(words, min_count):
            return {word:count for word, count in words.items() if count >= min_count}

        def as_bigram_tag(wordpos):
            poslist = [pos for _, pos in wordpos]
            return [(pos0, pos1) for pos0, pos1 in zip(poslist, poslist[1:])]

        pos2words = defaultdict(lambda: defaultdict(int))
        trans = defaultdict(int)
        bos = defaultdict(int)

        eos_tag = 'EOS'
        message_format = '\rtraining observation/transition prob from %d sents'

        for i, sent in enumerate(corpus):
            # generation prob
            for word, pos in sent:
                pos2words[pos][word] += 1
            # transition prob
            for bigram in as_bigram_tag(sent):
                trans[bigram] += 1
            # bos
            bos[sent[0][1]] += 1
            # eos
            trans[(sent[-1][1], eos_tag)] += 1
            if (self.verbose) and (i % 10000 == 0):
                print('%s ...'%(message_format%i), end='', flush=True)
        if self.verbose:
            print('%s was done'%(message_format%i), flush=True)

        pos2words = {pos:trim_words(words, self.min_count_word)
                     for pos, words in pos2words.items()}
        pos2words = {pos:words for pos, words in pos2words.items()
                     if sum(words.values()) >= self.min_count_tag}
        trans = {pos:count for pos, count in trans.items() if pos[0] in pos2words}
        bos = {pos:count for pos, count in bos.items() if pos in pos2words}

        return pos2words, trans, bos

    def _to_log_prob(self, pos2words, transition, bos):

        # observation
        base = {pos:sum(words.values()) for pos, words in pos2words.items()}
        pos2words_ = {pos:{word:math.log(count/base[pos]) for word, count in words.items()}
                      for pos, words in pos2words.items()}

        # transition
        base = defaultdict(int)
        for (pos0, pos1), count in transition.items():
            base[pos0] += count
        transition_ = {pos:math.log(count/base[pos[0]]) for pos, count in transition.items()}

        # bos
        base = sum(bos.values())
        bos_ = {pos:math.log(count/base) for pos, count in bos.items()}

        return pos2words_, transition_, bos_

    def _save_as_json(self, model_path, pos2words_=None, transition_=None, bos_=None):
        check_dirs(model_path)

        if not pos2words_:
            pos2words_ = self.pos2words_
        if not transition_:
            transition_ = self.transition_
        if not bos_:
            bos_ = self.bos_

        transition_json = {' '.join(pos):prob for pos, prob in transition_.items()}

        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(
                {'pos2words': pos2words_,
                 'transition': transition_json,
                 'bos': bos_
                },
                f, ensure_ascii=False, indent=2
            )