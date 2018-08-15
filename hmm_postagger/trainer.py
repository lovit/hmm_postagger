from collections import defaultdict

class CorpusTrainer:
    def __init__(self, tagset=None, min_count_tag=5, min_count_word=1, verbose=True):
        self.tagset = tagset
        self.min_count_tag = min_count_tag
        self.min_count_word = min_count_word
        self.verbose = verbose

    def train(self, corpus):
        pos2words = defaultdict(lambda: defaultdict(int))
        for i, sent in enumerate(corpus):
            for word, pos in sent:
                pos2words[pos][word] += 1
            if (self.verbose) and (i % 10000 == 0):
                print('\rtraining from %d sents ...'%i, end='', flush=True)
        if self.verbose:
            print('\rtraining from %d sents was done'%i)
        pos2words = {pos:dict(words) for pos, words in pos2words.items()}
        return pos2words