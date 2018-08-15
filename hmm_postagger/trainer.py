from collections import defaultdict

class CorpusTrainer:
    def __init__(self, tagset=None, min_count_tag=5, min_count_word=1, verbose=True):
        self.tagset = tagset
        self.min_count_tag = min_count_tag
        self.min_count_word = min_count_word
        self.verbose = verbose

    def train(self, corpus):
        pos2words = self._count_pos_words(corpus)
        transition = self._count_transition_prob(pos2words, corpus)
        return pos2words, transition

    def _count_pos_words(self, corpus):

        def trim_words(words, min_count):
            return {word:count for word, count in words.items() if count >= min_count}

        pos2words = defaultdict(lambda: defaultdict(int))
        for i, sent in enumerate(corpus):
            for word, pos in sent:
                pos2words[pos][word] += 1
            if (self.verbose) and (i % 10000 == 0):
                print('\rtraining observation prob from %d sents ...'%i, end='', flush=True)
        if self.verbose:
            print('\rtraining observation prob from %d sents was done'%i)
        pos2words = {pos:trim_words(words, self.min_count_word)
                     for pos, words in pos2words.items()}
        pos2words = {pos:words for pos, words in pos2words.items()
                     if sum(words.values()) >= self.min_count_tag}
        return pos2words

    def _count_transition_prob(self, tagset, corpus):

        def as_bigram_tag(wordpos):
            poslist = [pos for _, pos in wordpos]
            bigram = [(pos0, pos1) for pos0, pos1 in zip(poslist, poslist[1:])
                      if (pos0 in tagset) and (pos1 in tagset)]
            return bigram

        trans = defaultdict(int)
        for i, sent in enumerate(corpus):
            for bigram in as_bigram_tag(sent):
                trans[bigram] += 1
            if (self.verbose) and (i % 10000 == 0):
                print('\rtraining transition prob from %d sents ...'%i, end='', flush=True)
        if (self.verbose):
            print('\rtraining transition prob from %d sents was done'%i, flush=True)

        base = defaultdict(int)
        for (pos0, pos1), count in trans.items():
            base[pos0] += count
        trans = {pos:count/base[pos[0]] for pos, count in trans.items()}
        return trans
