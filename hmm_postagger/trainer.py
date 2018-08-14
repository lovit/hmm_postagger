class CorpusTrainer:
    def __init__(self, tagset=None, min_count_tag=5, min_count_word=1, verbose=True):
        self.tagset = tagset
        self.min_count_tag = min_count_tag
        self.min_count_word = min_count_word
        self.verbose = verbose

    def train(self, corpus):
        raise NotImplemented
