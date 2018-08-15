from collections import defaultdict
from .utils import check_dirs

class CorpusTrainer:
    def __init__(self, tagset=None, min_count_tag=5, min_count_word=1, verbose=True):
        self.tagset = tagset
        self.min_count_tag = min_count_tag
        self.min_count_word = min_count_word
        self.verbose = verbose

    def train(self, corpus, model_path=None):
        pos2words, transition = self._count_pos_words(corpus)
        self.pos2words_, self.transition_ = self._to_prob(pos2words, transition)
        if model_path:
            self._save_as_txt(model_path)

    def _count_pos_words(self, corpus):

        def trim_words(words, min_count):
            return {word:count for word, count in words.items() if count >= min_count}

        def as_bigram_tag(wordpos):
            poslist = [pos for _, pos in wordpos]
            return [(pos0, pos1) for pos0, pos1 in zip(poslist, poslist[1:])]

        pos2words = defaultdict(lambda: defaultdict(int))
        trans = defaultdict(int)

        message_format = '\rtraining observation/transition prob from %d sents'
        for i, sent in enumerate(corpus):
            # generation prob
            for word, pos in sent:
                pos2words[pos][word] += 1
            # transition prob
            for bigram in as_bigram_tag(sent):
                trans[bigram] += 1
            if (self.verbose) and (i % 10000 == 0):
                print('%s ...'%(message_format%i), end='', flush=True)
        if self.verbose:
            print('%s was done'%(message_format%i), flush=True)

        pos2words = {pos:trim_words(words, self.min_count_word)
                     for pos, words in pos2words.items()}
        pos2words = {pos:words for pos, words in pos2words.items()
                     if sum(words.values()) >= self.min_count_tag}
        trans = {pos:count for pos, count in trans.items() if pos[0] in pos2words}

        return pos2words, trans

    def _to_prob(self, pos2words, transition):

        # observation
        base = {pos:sum(words.values()) for pos, words in pos2words.items()}
        pos2words_ = {pos:{word:count/base[pos] for word, count in words.items()}
                      for pos, words in pos2words.items()}

        # transition
        base = defaultdict(int)
        for (pos0, pos1), count in transition.items():
            base[pos0] += count
        transition_ = {pos:count/base[pos[0]] for pos, count in transition.items()}

        return pos2words_, transition_

    def _save_as_txt(self, model_path, pos2words_=None, transition_=None):
        check_dirs(model_path)

        if not pos2words_:
            pos2words_ = self.pos2words_
        if not transition_:
            transition_ = self.transition_

        with open(model_path, 'w', encoding='utf-8') as f:
            f.write('##transition\n')
            for pos, prob in sorted(transition_.items()):
                f.write('{} {} {}\n'.format(pos[0], pos[1], prob))
            f.write('\n##generation')
            for pos, words in sorted(pos2words_.items()):
                f.write('\n#{}\n'.format(pos))
                for word, prob in sorted(words.items(), key=lambda x:-x[1]):
                    f.write('{} {}\n'.format(word, prob))