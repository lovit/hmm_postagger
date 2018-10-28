from collections import defaultdict
import json
import math

from .utils import check_dirs
from .utils import bos, eos

class CorpusTrainer:
    def __init__(self, tagset=None, min_count_tag=5, min_count_word=1, verbose=True):
        self.tagset = tagset
        self.min_count_tag = min_count_tag
        self.min_count_word = min_count_word
        self.verbose = verbose

    def train(self, corpus, model_path=None):

        emission, transition = self._count_pos_words(corpus)

        self.emission_, self.transition_ = self._to_log_prob(
            emission, transition)

        if model_path:
            if model_path[-4:] != 'json':
                model_path += '.json'
            self._save_as_json(model_path)

    def _count_pos_words(self, corpus):

        def trim_words(words, min_count):
            return {word:count for word, count in words.items() if count >= min_count}

        emission = defaultdict(lambda: defaultdict(int))
        transition = defaultdict(int)

        message_format = '\rtraining observation/transition prob from %d sents'

        for i, sent in enumerate(corpus):
            # generation prob
            for word, pos in sent:
                emission[pos][word] += 1
            tags = [bos] + [tag for word, tag in sent] + [eos]
            for t0, t1 in zip(tags, tags[1:]):
                bigram = (t0, t1)
                transition[bigram] += 1
            if (self.verbose) and (i % 10000 == 0):
                print('%s ...'%(message_format%i), end='', flush=True)
        if self.verbose:
            print('%s was done'%(message_format%i), flush=True)

        emission = {pos:trim_words(words, self.min_count_word)
                    for pos, words in emission.items()}
        emission = {pos:words for pos, words in emission.items()
                    if sum(words.values()) >= self.min_count_tag}
        transition = {pos:count for pos, count in transition.items() if pos[0] in emission}

        return emission, transition

    def _to_log_prob(self, emission, transition):

        # observation
        base = {pos:sum(words.values()) for pos, words in emission.items()}
        emission_ = {pos:{word:math.log(count/base[pos]) for word, count in words.items()}
                    for pos, words in emission.items()}

        # transition
        base = defaultdict(int)
        for (pos0, pos1), count in transition.items():
            base[pos0] += count
        transition_ = {pos:math.log(count/base[pos[0]]) for pos, count in transition.items()}

        return emission_, transition_

    def _save_as_json(self, model_path, emission_=None, transition_=None):
        check_dirs(model_path)

        if not emission_:
            emission_ = self.emission_
        if not transition_:
            transition_ = self.transition_

        transition_json = {' '.join(pos):prob for pos, prob in transition_.items()}

        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(
                {'emission': emission_,
                 'transition': transition_json
                },
                f, ensure_ascii=False, indent=2
            )